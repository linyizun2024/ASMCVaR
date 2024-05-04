# This is the experiment for the sparse relaxation test.
Please directly run `mainfile.m` to obtain the test results.

`LRdatagen`: function for generating data X and y.

`L0regress`: function for exhaustively enumerating support sets to obtain the globally optimal solution of the original model.

`RelaxL0regress`: function for exhaustively enumerating support sets to obtain the globally optimal solution of the relaxed model.

`PALMforRelaxL0regress`: function for iteratively solving the relaxed model by the PALM algorithm.

After running the mainfile, the variables `opt_beta_L0`, `opt_beta_RelaxL0`, and `beta_PALM` are generated, representing the results of invoking the `L0regress`, `RelaxL0regress`, and `PALMforRelaxL0regress` functions to obtain the beta vector. Similarly, `opt_supp_L0`, `opt_supp_RelaxL0`, and `final_supp_PALM` correspond to the support sets obtained from these functions. Additionally, `opt_loss_L0`, `opt_loss_RelaxL0`, and `final_loss_PALM` denote the final values of the loss function. By comparing the support sets, beta vectors, and loss function values from the `L0regress` and `RelaxL0regress` functions, we can assess the effectiveness of our relaxation strategy in obtaining the globally optimal solution of the original model. Similarly, comparing the results from the `RelaxL0regress` and `PALMforRelaxL0regress` functions allows us to determine if the PALM algorithm converges to the globally optimal solution of the relaxed model.
