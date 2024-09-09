import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from random import SystemRandom
import models
import utils
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--only-periodic', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--classify_pertp', action='store_true',
                    help="Include per time-point classification loss")
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Remove this line as it is specific to GPU
    # torch.cuda.manual_seed(seed)

    # Ensure the device is set to CPU
    device = torch.device('cpu')

    print("Loading dataset...")
    if args.dataset == 'toy':
        #data_obj = utils.get_toy_data(args)
        data_obj = utils.kernel_smoother_data_gen(args, alpha=100., seed=0)
    elif args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'airquality':
        data_obj = utils.get_air_quality_data(args, 'cpu', args.quantization)  # Updated to load Air Quality dataset
    #elif args.dataset == 'activity':
    #    data_obj = utils.get_activity_data(args, 'cpu')
    print("Dataset loaded.")

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    if 'val_dataloader' in data_obj:
        print("Validation data loaded.")
    else:
        val_loader = None
        print("No validation data available.")
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    os.makedirs('data_origin', exist_ok=True)
     # Extract and save original data before model training
#    print("Extracting and saving original data (T, X, M)...")
#    T_train_orig, X_train_orig, M_train_orig = extract_data_from_dataloader(train_loader, dim)
#    T_test_orig, X_test_orig, M_test_orig = extract_data_from_dataloader(test_loader, dim)
#    if val_loader is not None:
#        T_val_orig, X_val_orig, M_val_orig = extract_data_from_dataloader(val_loader, dim)
#        np.save('data_origin/T_val.npy', T_val_orig)
#        np.save('data_origin/X_val.npy', X_val_orig)
#        np.save('data_origin/M_val.npy', M_val_orig)
#        print("T_val shape:", T_val_orig.shape)
#        print("X_val shape:", X_val_orig.shape)
#        print("M_val shape:", M_val_orig.shape)

#    np.save('data_origin/T_train.npy', T_train_orig)
#    np.save('data_origin/X_train.npy', X_train_orig)
#    np.save('data_origin/M_train.npy', M_train_orig)
#    np.save('data_origin/T_test.npy', T_test_orig)
#    np.save('data_origin/X_test.npy', X_test_orig)
#    np.save('data_origin/M_test.npy', M_test_orig)

#    print("T_train_orig shape:", T_train_orig.shape)
#    print("X_train_orig shape:", X_train_orig.shape)
#    print("M_train_orig shape:", M_train_orig.shape)
#    print("T_test_orig shape:", T_test_orig.shape)
#    print("X_test_orig shape:", X_test_orig.shape)
#    print("M_test_orig shape:", M_test_orig.shape)

 #   print("done")

    # model
    print("Initializing models...")
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)
   
        
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)
    print("Models initialized.")

    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec))
    if args.fname is not None:
        print("Loading saved model weights...")
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))

    print("Starting training loop...")
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch in train_loader:
            print("Processing batch...")
            #print("train_batch content:", train_batch)
            data = train_batch[:, :, :-1]  # All features except the last one
            labels = train_batch[:, :, -2].squeeze()  # The last feature

            print(labels)
            data = data.to(device)
            labels = labels.to(device)
            print(data.shape)
            print(labels.shape)

            batch_len = train_batch.shape[0]
            print("train batch shape:", train_batch.shape)

            observed_data = data[:, :, :dim]
            observed_mask = data[:, :, dim:2 * dim]
            observed_tp = data[:, :, -1]


            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len

        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}'
            .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n))
        if itr % 10 == 0:
            print('Test Mean Squared Error', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        if itr % 10 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
            }, args.dataset + '_' + args.enc + '_' + args.dec + '_' +
                str(experiment_id) + '.h5')
    print("Training complete.")

    # Run the data through the trained model and capture the processed outputs
    def process_data_through_model(dataloader, model, decoder, dim):
        model.eval()
        all_data = []

        with torch.no_grad():
            for record_id, train_batch in enumerate(dataloader):
                batch_len = train_batch.shape[0]

                # Assuming that the last feature is the label
                data = train_batch[:, :, :-1]  # All features except the last one
                labels = train_batch[:, :, -2].squeeze()  # Extract all labels (assuming label is the last feature)

                # Move data and labels to the appropriate device
                if labels is not None and len(labels) > 0:
                    data = data.to(device)
                    labels = labels.to(device)

                    observed_data = data[:, :, :dim]
                    observed_mask = data[:, :, dim:2 * dim]
                    observed_tp = data[:, :, -1]

                    out = model(torch.cat((observed_data, observed_mask), 2), observed_tp)
                    qz0_mean = out[:, :, :args.latent_dim]
                    qz0_logvar = out[:, :, args.latent_dim:]
                    epsilon = torch.randn(
                        1, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
                    ).to(device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                    z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])

                    pred_x = decoder(
                        z0,
                        observed_tp[None, :, :].repeat(1, 1, 1).view(-1, observed_tp.shape[1])
                    )
                    pred_x = pred_x.view(1, observed_data.shape[0], observed_data.shape[1], observed_data.shape[2]).mean(0)

                    # Append structured data
                    for i in range(batch_len):
                        record = (record_id, observed_tp[i].cpu().numpy(), observed_data[i].cpu().numpy(), observed_mask[i].cpu().numpy(), labels[i].cpu().numpy())
                        all_data.append(record)

        return all_data


    train_data = process_data_through_model(train_loader, rec, dec, dim)
    test_data = process_data_through_model(test_loader, rec, dec, dim)
    val_data = process_data_through_model(val_loader, rec, dec, dim) if val_loader else None

    # Save the data
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    if val_data:
        with open('val_data.pkl', 'wb') as f:
            pickle.dump(val_data, f)

    print("Data saved as train_data.pkl, test_data.pkl, and val_data.pkl.")