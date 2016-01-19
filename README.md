
# Wasserstein Loss Layer
Implements the Wasserstein Loss Layer (as described in [Frogner et al. 2015](http://cbcl.mit.edu/wasserstein/)) in Caffe. (Previously, the only
open source implementation of the loss function was in
Mocha.jl, which we used as a reference)
## Project Report
Check 	./project.pdf
## Example Models using the loss layer
Check 	./examples/mnist and 	./examples/sample_wasserstein
## TODO
Fix the test to check for difference in gradient.