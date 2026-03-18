output "public_ip"  { value = aws_instance.jenkins.public_ip }
output "public_dns" { value = aws_instance.jenkins.public_dns }
output "ssh_command" {
  value = "ssh -i ${path.root}/jenkins-key.pem ubuntu@${aws_instance.jenkins.public_ip}"
}
