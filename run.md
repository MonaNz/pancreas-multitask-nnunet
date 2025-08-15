# Quick test (20 epochs)
python working_standalone.py --mode train --epochs 20 --thermal_protection

# Full training
python working_standalone.py --mode train --epochs 100 --thermal_protection

# Complete pipeline
python working_standalone.py --mode all --epochs 50 --thermal_protection

# Just inference (after training)
python working_standalone.py --mode inference