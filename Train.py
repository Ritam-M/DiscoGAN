def Train():
  prev_time = time.time()
  for epoch in range(opt.epoch, opt.n_epochs):
      for i, batch in enumerate(dataloader):

          # Model inputs
          real_A = Variable(batch["A"].type(Tensor))
          real_B = Variable(batch["B"].type(Tensor))

          # Adversarial ground truths
          valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
          fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

          # ------------------
          #  Train Generators
          # ------------------

          G_AB.train()
          G_BA.train()

          optimizer_G.zero_grad()

          # GAN loss
          fake_B = G_AB(real_A)
          loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
          fake_A = G_BA(real_B)
          loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)

          loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

          # Pixelwise translation loss
          loss_pixelwise = (pixelwise_loss(fake_A, real_A) + pixelwise_loss(fake_B, real_B)) / 2

          # Cycle loss
          loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
          loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
          loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

          # Total loss
          loss_G = loss_GAN + loss_cycle + loss_pixelwise

          loss_G.backward()
          optimizer_G.step()

          # -----------------------
          #  Train Discriminator A
          # -----------------------

          optimizer_D_A.zero_grad()

          # Real loss
          loss_real = adversarial_loss(D_A(real_A), valid)
          # Fake loss (on batch of previously generated samples)
          loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)
          # Total loss
          loss_D_A = (loss_real + loss_fake) / 2

          loss_D_A.backward()
          optimizer_D_A.step()

          # -----------------------
          #  Train Discriminator B
          # -----------------------

          optimizer_D_B.zero_grad()
          # Real loss
          loss_real = adversarial_loss(D_B(real_B), valid)
          # Fake loss (on batch of previously generated samples)
          loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)
          # Total loss
          loss_D_B = (loss_real + loss_fake) / 2

          loss_D_B.backward()
          optimizer_D_B.step()

          loss_D = 0.5 * (loss_D_A + loss_D_B)

          # --------------
          #  Log Progress
          # --------------

          # Determine approximate time left
          batches_done = epoch * len(dataloader) + i
          batches_left = opt.n_epochs * len(dataloader) - batches_done
          time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
          prev_time = time.time()

          # Print log
          sys.stdout.write(
              "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s"
              % (
                  epoch,
                  opt.n_epochs,
                  i,
                  len(dataloader),
                  loss_D.item(),
                  loss_G.item(),
                  loss_GAN.item(),
                  loss_pixelwise.item(),
                  loss_cycle.item(),
                  time_left,
              )
          )

          # If at sample interval save image
          if batches_done % opt.sample_interval == 0:
              sample_images(batches_done)

      if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
          # Save model checkpoints
          torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
          torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
          torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
          torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
