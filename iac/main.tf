terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "us-east-1"
  profile = "atn-developer"
}

resource "aws_elastic_beanstalk_application" "eb_app" {
  name        = "getout-of-text3"
  description = "getout-of-text3 application"
}

resource "aws_elastic_beanstalk_environment" "eb_app_env" {
  name                = "etienne-app-env"
  application         = aws_elastic_beanstalk_application.eb_app.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.5.2 running Python 3.11"
  
  setting {
    namespace   = "aws:autoscaling:launchconfiguration"
    name        = "IamInstanceProfile"
    value       = "aws-elasticbeanstalk-ec2-role"
  }
  setting {
    namespace   = "aws:autoscaling:launchconfiguration"
    name        = "DisableIMDSv1"
    value       = "true"
  }
  setting { 
    namespace   = "aws:autoscaling:asg"
    name        = "MinSize"
    value       = "1"
  }
  setting {
    namespace   = "aws:autoscaling:asg"
    name        = "MaxSize"
    value       = "1"
  }
  setting {
    namespace   = "aws:elasticbeanstalk:environment"
    name        = "EnvironmentType"
    value       = "LoadBalanced"
  }
  setting {
    namespace   = "aws:elasticbeanstalk:environment"
    name        = "LoadBalancerType"
    value       = "application"
  }

  setting {
    namespace   = "aws:ec2:instances"
    name        = "InstanceTypes"
    value       = "t4g.small"
  }

  setting {
    namespace   = "aws:elasticbeanstalk:environment:proxy"
    name        = "ProxyServer"
    value       = "apache"
  }
}

# Reference the existing hosted zone for ejacquot.com
data "aws_route53_zone" "ejacquot" {
  name         = "ejacquot.com"
  private_zone = false
}

resource "aws_route53_record" "eb_dns" {
  zone_id = data.aws_route53_zone.ejacquot.id
  name    = "getout-of-text3.ejacquot.com"
  type    = "CNAME"
  ttl     = "300"
  records = [aws_elastic_beanstalk_environment.eb_app_env.endpoint_url]
}