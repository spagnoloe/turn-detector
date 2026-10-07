# Laptop-only compute: small frozen encoders, few hours of data

All data preparation, training and serving run on a single 24 GB MacBook Air, with no cloud GPU. We therefore use small pretrained encoders kept frozen, train only lightweight classifier heads, and work with a few hours of audio rather than the full 104 h corpus. Accuracy is traded for a pipeline that runs end-to-end on one machine within the assignment's time budget.
