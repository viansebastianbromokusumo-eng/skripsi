

def compute_total_loss(
    model,
    low_light_input,
    high_light_target,
    detection_targets,
    det_loss_fn,
    enhance_loss_fn,
    inference=False,
    det_loss_w=0.6,
    enh_loss_w=0.4,
):

    # Forward Pass
    # det_out = list of tensors from detection head
    # llie_out = enhanced image tensor (R)
    det_out, llie_out, llie_residuals = model(low_light_input, inference=inference)

    # Calculate Detection Loss (Loss_Det)
    # det_loss_fn is an instance of ComputeLoss (self)
    
    # loss_box, loss_cls, loss_dfl = det_loss_fn(det_out, detection_targets)
    # det_loss_components = loss_box + loss_cls + loss_dfl
    
    det_loss = det_loss_fn(detection_targets, preds=det_out)
    det_loss_components = det_loss

    # Calculate Enhancement Loss (Loss_Enh)
    # enhance_loss_fn is the MSE, MAE, or SmoothL1 function
    # Note: Assumes llie_out and high_light_target have the same shape (B, C, H, W)
    loss_enhancement = enhance_loss_fn(llie_out, high_light_target)

    # Total Loss = 0.6 * Loss_Det + 0.4 * Loss_Enh
    total_loss = (det_loss_w * det_loss_components) + (enh_loss_w * loss_enhancement)

    return total_loss, det_loss_components, loss_enhancement, det_out, llie_out, llie_residuals


def dynamic_total_loss(
    model,
    low_light_input,
    high_light_target,
    detection_targets,
    det_loss_fn,
    enhance_loss_fn,
    dynamic_weighter, 
    inference=False,
):
    # Forward Pass
    det_out, llie_out, llie_residuals = model(low_light_input, inference=inference)

    # Calculate Raw Detection Loss
    det_loss_raw = det_loss_fn(detection_targets, preds=det_out)

    # Calculate Raw Enhancement Loss
    loss_enhancement_raw = enhance_loss_fn(llie_out, high_light_target)

    # Apply Dynamic Weighting
    # calls the forward() of DynamicLossWeighting
    total_loss, prec_det, prec_enh = dynamic_weighter(det_loss_raw, loss_enhancement_raw)

    return total_loss, det_loss_raw, loss_enhancement_raw, det_out, llie_out, llie_residuals, prec_det, prec_enh



def compute_total_loss_unsup(
    model,
    low_light_input,
    detection_targets,
    det_loss_fn,
    enhance_loss_fn,
    det_loss_w=0.2,
    enh_loss_w=10.0,
):

    # 1. Forward Pass
    # det_out = list of tensors from detection head
    # llie_out = enhanced image tensor (R)
    det_out, llie_out, llie_residuals = model(low_light_input, inference=False)

    # 2. Calculate Detection Loss (Loss_Det)
    # det_loss_fn is an instance of ComputeLoss (self)
    
    # loss_box, loss_cls, loss_dfl = det_loss_fn(det_out, detection_targets)
    # det_loss_components = loss_box + loss_cls + loss_dfl
    
    det_loss = det_loss_fn(detection_targets, preds=det_out)
    det_loss_components = det_loss

    # 3. Calculate Enhancement Loss (Loss_Enh)
    # enhance_loss_fn is the MSE, MAE, or SmoothL1 function
    # Note: Assumes llie_out and high_light_target have the same shape (B, C, H, W)
    loss_enhancement = enhance_loss_fn(llie_out)

    # Total Loss = 0.6 * Loss_Det + 0.4 * Loss_Enh
    total_loss = (det_loss_w * det_loss_components) + (enh_loss_w * loss_enhancement)

    return total_loss, det_loss_components, loss_enhancement, det_out, llie_out, llie_residuals