import re
import matplotlib.pyplot as plt
import pandas as pd

#loss_filename = "losses_1575458071.3440142.txt"
loss_filename = "losses_vgg_1575624281.130714.txt"

rgx = r'\[[0-9.]*\]'

disc_loss = []
gen_loss = []

with open(loss_filename, "r+") as f:
  for line in f.readlines():
    try:
      nums = re.findall(rgx, line)
      disc = float(nums[2][1:-1])
      gen = float(nums[3][1:-1])
      disc_loss.append(disc)
      gen_loss.append(gen)
    except IndexError:
      pass

#df=pd.DataFrame({'x_disc': range(len(disc_loss)), 'x_gen': range(len(gen_loss)), 'disc_loss': disc_loss, 'gen_loss': gen_loss})

#plt.plot('x_disc', 'disc_loss', data=df, marker='', color='blue', linewidth=2)
#plt.plot('x_gen', 'gen_loss', data=df, marker='', color='orange', linewidth=2)
#plt.legend()
plt.plot(range(len(disc_loss)),disc_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Discriminator Model Loss')
plt.legend(['disc loss'])
plt.savefig(loss_filename[:-4]+'_discriminator'+'.png')
print('Saved figure at {}'.format(loss_filename[:-4]+'_discriminator'+'.png'))
plt.close()

plt.plot(range(len(gen_loss)), gen_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Generator Model Loss')
plt.legend(['gen loss'])
plt.savefig(loss_filename[:-4]+'_generator'+'.png')
print('Saved figure at {}'.format(loss_filename[:-4]+'_generator'+'.png'))
plt.close()
 
