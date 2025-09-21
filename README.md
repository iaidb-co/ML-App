# 🏠 Machine Learning House Price Prediction on AWS

A complete machine learning application deployed on AWS EC2 with Streamlit, using S3 for data storage and model persistence.

## 🚀 Live Demo
**Deployed Application**: `http://your-ec2-public-ip:8501`

## 📁 Project Structure
```
ml-streamlit-aws/
├── app.py                 # Main Streamlit application
├── model.py              # ML model training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── setup_scripts/       # Deployment scripts
│   ├── ec2_setup.sh     # EC2 initialization script
│   └── deploy.sh        # Deployment script
├── docs/                # Documentation
│   ├── aws_setup.md     # AWS setup guide
│   └── architecture.md  # System architecture
└── data/               # Local data directory (optional)
    └── sample_data.csv # Sample dataset
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Users/Web     │    │   EC2 Instance  │    │   S3 Bucket     │
│   Interface     │◄──►│   (t2.micro)    │◄──►│   Data Storage  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌──────▼──────┐
                       │  Streamlit  │
                       │ Application │
                       └─────────────┘
                              │
                       ┌──────▼──────┐
                       │ ML Pipeline │
                       │ (Scikit-learn)│
                       └─────────────┘
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-learn
- **Cloud Infrastructure**: AWS EC2, S3, IAM
- **ML Framework**: Random Forest Regression
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Ubuntu Server, systemd

## 📋 Prerequisites

1. **AWS Account** with Free Tier access
2. **Python 3.8+** installed locally
3. **AWS CLI** configured
4. **Git** for version control
5. **Basic knowledge** of Linux/terminal commands

## 🚦 Quick Start Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/ml-streamlit-aws.git
cd ml-streamlit-aws
```

### Step 2: Set Up AWS CLI
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Access Key, Region (us-west-2), and output format (json)
```

### Step 3: Create AWS Resources
```bash
# Create S3 bucket (replace with unique name)
aws s3 mb s3://ml-house-prediction-bucket-2024

# Create EC2 key pair
aws ec2 create-key-pair --key-name ml-deployment-key --query 'KeyMaterial' --output text > ml-deployment-key.pem
chmod 400 ml-deployment-key.pem
```

### Step 4: Launch EC2 Instance
```bash
# Create security group
aws ec2 create-security-group --group-name ml-security-group --description "Security group for ML app"

# Add rules to security group
aws ec2 authorize-security-group-ingress --group-name ml-security-group --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name ml-security-group --protocol tcp --port 8501 --cidr 0.0.0.0/0

# Launch instance (replace ami-id with your region's Ubuntu AMI)
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type t2.micro \
    --key-name ml-deployment-key \
    --security-groups ml-security-group \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ML-Streamlit-App}]'
```

### Step 5: Deploy Application on EC2
```bash
# Get your instance's public IP
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' --output table

# SSH into your instance
ssh -i ml-deployment-key.pem ubuntu@your-ec2-public-ip

# On EC2 instance:
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip git -y
git clone https://github.com/yourusername/ml-streamlit-aws.git
cd ml-streamlit-aws
pip3 install -r requirements.txt

# Configure AWS CLI on EC2 (or use IAM roles - recommended)
aws configure

# Run the application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔧 Detailed Setup Instructions

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### AWS IAM Role Setup (Recommended)
Instead of using access keys on EC2, create an IAM role:

1. **Create IAM Role**:
   ```bash
   aws iam create-role --role-name EC2-S3-Access-Role --assume-role-policy-document '{
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {"Service": "ec2.amazonaws.com"},
         "Action": "sts:AssumeRole"
       }
     ]
   }'
   ```

2. **Attach Policies**:
   ```bash
   aws iam attach-role-policy --role-name EC2-S3-Access-Role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
   ```

3. **Create Instance Profile**:
   ```bash
   aws iam create-instance-profile --instance-profile-name EC2-S3-Profile
   aws iam add-role-to-instance-profile --instance-profile-name EC2-S3-Profile --role-name EC2-S3-Access-Role
   ```

### Production Deployment with systemd
Create service file on EC2:

```bash
sudo nano /etc/systemd/system/streamlit-app.service
```

Content:
```ini
[Unit]
Description=Streamlit ML Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ml-streamlit-aws
Environment=PATH=/home/ubuntu/.local/bin
ExecStart=/home/ubuntu/.local/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app
sudo systemctl status streamlit-app
```

## 📊 Features

### 🏠 House Price Prediction
- **Input Features**: Bedrooms, bathrooms, square footage, lot size, floors, waterfront, condition, grade, year built
- **Model**: Random Forest Regression with feature scaling
- **Performance Metrics**: R², MSE, RMSE, MAE

### 📈 Data Visualization
- Interactive price distribution histograms
- Feature correlation heatmaps
- Actual vs. predicted price scatter plots
- Feature importance analysis
- Residuals analysis

### ☁️ AWS Integration
- **S3 Storage**: Dataset and model persistence
- **EC2 Deployment**: Scalable web application hosting
- **IAM Security**: Role-based access control

## 🔒 Security Considerations

1. **Network Security**: Properly configured security groups
2. **Access Control**: IAM roles instead of hardcoded credentials
3. **Data Privacy**: Encrypted S3 buckets (optional)
4. **SSH Access**: Key-based authentication only

## 💰 Cost Management

- **Free Tier Usage**: t2.micro EC2 instance (750 hours/month free)
- **S3 Costs**: 5GB storage free per month
- **Data Transfer**: 1GB outbound free per month
- **Estimated Monthly Cost**: $0-5 within Free Tier limits

## 📈 Performance Optimization

### Model Performance
- **Training Time**: ~30 seconds on t2.micro
- **Inference Time**: <100ms per prediction
- **Model Size**: ~2MB (Random Forest with 100 trees)
- **Memory Usage**: ~500MB peak during training

### Application Performance
- **Load Time**: ~3-5 seconds
- **Concurrent Users**: 10-20 on t2.micro
- **Response Time**: <1 second for predictions

## 🐛 Troubleshooting

### Common Issues

1. **Port 8501 not accessible**
   ```bash
   # Check security group
   aws ec2 describe-security-groups --group-names ml-security-group
   
   # Check if app is running
   sudo netstat -tlnp | grep 8501
   ```

2. **S3 access denied**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Test S3 access
   aws s3 ls s3://your-bucket-name
   ```

3. **Memory issues on t2.micro**
   ```bash
   # Monitor memory usage
   free -h
   htop
   
   # Add swap file if needed
   sudo fallocate -l 1G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Application crashes**
   ```bash
   # Check logs
   journalctl -u streamlit-app -f
   
   # Restart service
   sudo systemctl restart streamlit-app
   ```

## 📚 Additional Resources

- [AWS Free Tier Guide](https://aws.amazon.com/free/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- AWS for providing Free Tier resources
- Streamlit for the amazing framework
- Scikit-learn for ML capabilities
- The open-source community for tools and inspiration

---

**📞 Need Help?**
- Create an issue in this repository
- Check the [troubleshooting section](#-troubleshooting)
- Review AWS documentation
- Join the Streamlit community forum
