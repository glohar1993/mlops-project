variable "project"            { type = string }
variable "environment"        { type = string }
variable "vpc_id"             { type = string }
variable "public_subnet_ids"  { type = list(string) }
variable "node_instance_type" { type = string }
variable "node_min"           { type = number }
variable "node_max"           { type = number }
