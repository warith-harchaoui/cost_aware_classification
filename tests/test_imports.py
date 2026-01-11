def test_imports():
    import cost_aware_losses
    from cost_aware_losses import SinkhornFenchelYoungLoss, SinkhornEnvelopeLoss, SinkhornFullAutodiffLoss

    assert SinkhornFenchelYoungLoss is not None
    assert SinkhornEnvelopeLoss is not None
    assert SinkhornFullAutodiffLoss is not None
