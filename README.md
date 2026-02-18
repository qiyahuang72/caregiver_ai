# Guardian: Privacy-First AI Caregiver
**Developed for the Dell/NVIDIA Hackathon 2026**

Aegis Guardian leverages edge AI to provide real-time safety monitoring for the elderly without compromising their privacy.

##  Key Features
- **Fall Detection:** Real-time spatial analysis of human biomechanics.
- **Privacy Mode:** Processes raw video at the edge and displays only anonymized "skeleton" data.
- **Hardware Accelerated:** Optimized for the NVIDIA GB10 (Blackwell) architecture.

##  Tech Stack
- **Hardware:** Dell Pro Max 10GB | NVIDIA GB10 GPU
- **AI Models:** YOLOv8-Pose (Medium)
- **Library:** PyTorch (CUDA 12.1), OpenCV, Ultralytics

##  Performance Benchmarks (GB10)
- **Inference Latency:** 
- **Throughput:**
- **Precision:** 

##  Installation & Setup
1. Clone the repo: `git clone https://github.com/qiyahuang72/caregiver_ai.git`
2. Enter directory: `cd caregiver_ai`
3. Setup Venv: `python3 -m venv venv && source venv/bin/activate`
4. Install Deps: `pip install -r requirements.txt`
5. Run Demo: `python guardian.py`
