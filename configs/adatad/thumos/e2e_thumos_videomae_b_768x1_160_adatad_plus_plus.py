_base_ = ["e2e_thumos_videomae_b_768x1_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            adapter_conv_type="adatad_plus_plus",  # Use the new adapter type
            adapter_deformable_groups=2,  # Default, but explicit here
        ),
    ),
)

# Using the standard AdaTAD config structure
optimizer = dict(
    backbone=dict(
        custom=[
            dict(
                name="adapter", lr=2e-4, weight_decay=0.05
            ),  # Keeping standard AdaTAD lr as baseline
        ]
    )
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_b_768x1_160_adatad_plus_plus"
