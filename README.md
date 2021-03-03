# mcvae
test mcvae

Code from https://github.com/diazandr3s/MI_pred_mcvae_ukbb

Try to reproduce the training part of the model, in main_mcVAE.py
Missing input_data/ids folder, modify input data then encountered the following problem:

  File "C:\Users\scnc\Downloads\code\MI_pred_mcvae_ukbb-master\main_mcVAE.py", line 124
    loaded_model = utilities.load_model(args.dir_results + args.dir_te st_ids)
                                                                            ^
SyntaxError: invalid syntax

The dir_results related part of the result in the code seems to need to be modified
