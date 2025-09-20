# 🚀 Complete GPU Training Setup Guide

This guide will help you set up a fully functional GPU training environment for Signature AI on AWS.

## 📋 Prerequisites

- AWS Account with billing enabled
- AWS CLI installed and configured
- Basic understanding of AWS EC2 and S3

## 🔧 Step 1: AWS Infrastructure Setup

### 1.1 Create S3 Bucket
```bash
# Replace 'your-unique-bucket-name' with a unique name
aws s3 mb s3://your-unique-bucket-name --region us-east-1
```

### 1.2 Create IAM Role for EC2
```bash
# Create the role
aws iam create-role --role-name EC2-S3-Access --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach S3 permissions
aws iam attach-role-policy --role-name EC2-S3-Access --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-S3-Access
aws iam add-role-to-instance-profile --instance-profile-name EC2-S3-Access --role-name EC2-S3-Access
```

### 1.3 Create Security Group
```bash
# Create security group
aws ec2 create-security-group --group-name gpu-training-sg --description "Security group for GPU training instances"

# Allow SSH access (replace with your IP for security)
aws ec2 authorize-security-group-ingress --group-name gpu-training-sg --protocol tcp --port 22 --cidr 0.0.0.0/0

# Allow HTTP/HTTPS
aws ec2 authorize-security-group-ingress --group-name gpu-training-sg --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name gpu-training-sg --protocol tcp --port 443 --cidr 0.0.0.0/0
```

### 1.4 Create Key Pair
```bash
# Create key pair for SSH access
aws ec2 create-key-pair --key-name gpu-training-key --query 'KeyMaterial' --output text > gpu-training-key.pem
chmod 400 gpu-training-key.pem
```

### 1.5 Get Resource IDs
```bash
# Get Security Group ID
aws ec2 describe-security-groups --group-names gpu-training-sg --query 'SecurityGroups[0].GroupId' --output text

# Get Subnet ID
aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text
```

## ⚙️ Step 2: Configure Your Application

### 2.1 Update .env File
Create a `.env` file with your configuration:

```bash
# Copy the example file
cp .env.example .env

# Edit with your values
nano .env
```

Required values:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET=your-unique-bucket-name

# GPU Training Configuration
AWS_KEY_NAME=gpu-training-key
AWS_SECURITY_GROUP_ID=sg-xxxxxxxxx  # From step 1.5
AWS_SUBNET_ID=subnet-xxxxxxxxx      # From step 1.5

# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## 🚀 Step 3: Deploy Training Code

### 3.1 Upload Code to S3
```bash
# Create a deployment package
cd /workspace/ai-training
zip -r training-code.zip . -x "*.git*" "*.pyc" "__pycache__/*" "*.log"

# Upload to S3
aws s3 cp training-code.zip s3://your-bucket/training-code.zip
```

### 3.2 Test GPU Training
```bash
# Start your local API server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# In another terminal, test GPU training
curl -X POST "http://localhost:8000/api/training/start-gpu-training" \
  -F "student_id=123" \
  -F "use_gpu=true" \
  -F "genuine_files=@signature1.jpg" \
  -F "genuine_files=@signature2.jpg"
```

## 🔍 Step 4: Monitor Training

### 4.1 Check Instance Status
```bash
# List running GPU instances
aws ec2 describe-instances --filters "Name=tag:Purpose,Values=AI-Training" --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' --output table
```

### 4.2 View Training Logs
```bash
# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances --filters "Name=tag:Purpose,Values=AI-Training" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# SSH into instance
ssh -i gpu-training-key.pem ubuntu@$INSTANCE_IP

# View training logs
tail -f /var/log/ai-training/*.log
```

### 4.3 Monitor Progress
```bash
# Check training progress via API
curl "http://localhost:8000/api/progress/stream/{job_id}"
```

## 🛠️ Step 5: Troubleshooting

### Common Issues and Solutions

#### 1. Instance Launch Fails
**Problem**: Instance fails to launch
**Solutions**:
- Check IAM permissions
- Verify security group settings
- Ensure key pair exists
- Check instance limits

#### 2. Training Fails
**Problem**: Training job fails
**Solutions**:
- Check S3 bucket permissions
- Verify training data format
- Monitor instance logs
- Check GPU availability

#### 3. High Costs
**Problem**: Unexpected high costs
**Solutions**:
- Set up billing alerts
- Use smaller instance types
- Monitor instance termination
- Use Spot instances for 90% savings

#### 4. No Training Logs
**Problem**: No progress updates
**Solutions**:
- Check frontend connection
- Verify job ID
- Check API endpoints
- Monitor instance status

### Debug Commands

```bash
# Check GPU availability
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Test S3 connectivity
aws s3 ls s3://your-bucket/

# Check instance health
python3 /home/ubuntu/ai-training/health_check.py

# Monitor system resources
python3 /home/ubuntu/ai-training/monitor.py
```

## 💰 Cost Optimization

### Instance Types and Costs

| Instance Type | GPU | vCPUs | RAM | Cost/Hour | Training Time | Total Cost |
|---------------|-----|-------|-----|-----------|---------------|------------|
| g4dn.xlarge   | 1x T4 | 4 | 16GB | $0.526 | ~15-30 min | $0.13-0.26 |
| g4dn.2xlarge  | 1x T4 | 8 | 32GB | $0.752 | ~10-20 min | $0.13-0.25 |
| p3.2xlarge    | 1x V100 | 8 | 61GB | $3.06 | ~5-10 min | $0.26-0.51 |

### Cost-Saving Tips

1. **Use Spot Instances**: Up to 90% cost savings
2. **Set Billing Alerts**: Monitor costs in real-time
3. **Auto-terminate**: Instances terminate after training
4. **Right-size**: Use appropriate instance types
5. **Monitor Usage**: Track with AWS Cost Explorer

## 📊 Performance Monitoring

### Real-time Metrics

The training system provides real-time metrics:

- **Progress Updates**: 0% → 100% completion
- **Training Stage**: Preprocessing → Training → Saving
- **Epoch Progress**: Epoch 1/50, 2/50, etc.
- **Loss and Accuracy**: Live training metrics
- **Time Estimates**: Remaining training time

### Monitoring Tools

```bash
# System monitoring
python3 /home/ubuntu/ai-training/monitor.py

# GPU health check
python3 /home/ubuntu/ai-training/health_check.py

# View logs
tail -f /var/log/ai-training/*.log

# Check training results
aws s3 ls s3://your-bucket/training_results/
```

## 🎯 Quick Start Checklist

- [ ] AWS account with billing enabled
- [ ] AWS CLI installed and configured
- [ ] S3 bucket created
- [ ] IAM role created
- [ ] Security group created
- [ ] Key pair created
- [ ] Subnet ID obtained
- [ ] `.env` file configured
- [ ] Training code uploaded
- [ ] API server started
- [ ] Test training completed

## 🚀 Advanced Configuration

### Custom AMI
Create a custom AMI with pre-installed dependencies:

```bash
# Launch instance with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.xlarge \
  --key-name gpu-training-key \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx

# Install additional dependencies
# ... (run setup_gpu_instance.sh)

# Create AMI
aws ec2 create-image \
  --instance-id i-1234567890abcdef0 \
  --name "signature-ai-gpu-training" \
  --description "Pre-configured GPU instance for signature AI training"
```

### Auto-scaling
Set up auto-scaling for multiple training jobs:

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name signature-ai-gpu \
  --launch-template-data '{
    "ImageId": "ami-0c02fb55956c7d316",
    "InstanceType": "g4dn.xlarge",
    "KeyName": "gpu-training-key",
    "SecurityGroupIds": ["sg-xxxxxxxxx"],
    "SubnetId": "subnet-xxxxxxxxx",
    "IamInstanceProfile": {"Name": "EC2-S3-Access"},
    "UserData": "'$(base64 -w 0 setup_gpu_instance.sh)'"
  }'
```

## 📞 Support

If you encounter issues:

1. **Check Logs**: Review training and system logs
2. **Verify Configuration**: Ensure all settings are correct
3. **Test Components**: Verify each component individually
4. **Monitor Resources**: Check CPU, memory, and GPU usage
5. **Review Documentation**: Check AWS and TensorFlow docs

## 🎉 Success!

Once you complete these steps, your GPU training system will be:

- ✅ **10-50x faster** than CPU training
- ✅ **Real-time progress** updates
- ✅ **Cost-effective** at $0.13-0.26 per session
- ✅ **Automatically scalable**
- ✅ **Production-ready**

**Your AI training is now ready for production use!** 🚀