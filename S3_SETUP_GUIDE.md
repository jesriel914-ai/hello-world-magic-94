# S3 Setup Guide for Signature AI Training

This guide will help you fix the "Failed to fetch" error when exporting models to S3.

## The Problem

The error occurs because AWS S3 credentials are not configured. The application tries to upload models to S3 but fails due to missing environment variables.

## Quick Fix

### Option 1: Automated Setup (Recommended)

1. Run the setup script:
   ```bash
   node setup-s3.js
   ```

2. Follow the prompts to enter your AWS credentials

3. Restart your development server:
   ```bash
   npm run dev
   ```

### Option 2: Manual Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your AWS credentials:
   ```env
   NEXT_PUBLIC_AWS_ACCESS_KEY_ID=your_aws_access_key_here
   NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
   NEXT_PUBLIC_AWS_REGION=us-east-1
   NEXT_PUBLIC_S3_BUCKET=signatureai-uploads
   ```

3. Restart your development server

## AWS Setup Requirements

### 1. Create AWS Account
- Sign up at [aws.amazon.com](https://aws.amazon.com)
- Complete account verification

### 2. Create S3 Bucket
1. Go to AWS S3 Console
2. Click "Create bucket"
3. Choose a unique bucket name (e.g., `your-company-signatureai-uploads`)
4. Select your preferred region
5. Leave other settings as default
6. Click "Create bucket"

### 3. Create IAM User
1. Go to AWS IAM Console
2. Click "Users" → "Create user"
3. Enter username: `signatureai-s3-user`
4. Select "Programmatic access"
5. Attach policy: `AmazonS3FullAccess` (or create custom policy)
6. Save the Access Key ID and Secret Access Key

### 4. Configure Bucket Permissions
1. Go to your S3 bucket
2. Click "Permissions" tab
3. Add this bucket policy (replace `YOUR-BUCKET-NAME`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowSignatureAIUploads",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::YOUR-ACCOUNT-ID:user/signatureai-s3-user"
            },
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
        }
    ]
}
```

## Testing the Fix

1. Start your development server:
   ```bash
   npm run dev
   ```

2. Open the application in your browser

3. Train a model with some signature images

4. Try to export the model to S3

5. You should see success messages instead of "Failed to fetch"

## Troubleshooting

### Still Getting "Failed to fetch"?
1. Check that your `.env` file exists and has the correct variables
2. Restart your development server after changing `.env`
3. Verify your AWS credentials are correct
4. Check that your S3 bucket exists and is accessible

### S3 Permission Errors?
1. Verify your IAM user has S3 permissions
2. Check your bucket policy allows the IAM user
3. Ensure the bucket name in `.env` matches your actual bucket

### Environment Variables Not Loading?
1. Make sure `.env` is in the project root directory
2. Restart your development server completely
3. Check for typos in variable names (they must start with `NEXT_PUBLIC_`)

## Alternative: Use Local Downloads Only

If you don't want to use S3, you can disable S3 exports:

1. Set in your `.env` file:
   ```env
   VITE_ENABLE_S3_STORAGE=false
   ```

2. Restart your development server

3. The export button will download models locally instead of uploading to S3

## Security Notes

- Never commit your `.env` file to version control
- Use IAM users with minimal required permissions
- Consider using AWS IAM roles for production deployments
- Rotate your AWS keys regularly

## Support

If you're still having issues:
1. Check the browser console for detailed error messages
2. Look at the server logs for S3-related errors
3. Verify your AWS account has the necessary permissions
4. Test your S3 credentials using AWS CLI: `aws s3 ls s3://your-bucket-name`