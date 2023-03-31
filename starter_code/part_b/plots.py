import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'C:\Users\Cherry\Desktop\School\CSC311\Project\csc311\starter_code\part_b\log.csv')

lamb = df['lambda'].unique()
learning_rate = df['lr'].unique()
step = df['step'].unique()
plots = ['val_acc', 'train_loss', 'test_acc']

# for l in lamb:
#     for lr in learning_rate:
#         for s in step:
#             for p in plots:
#                 rows = df.loc[(df['lambda'] == l) & (df['lr'] == lr) & (df['step'] == s)]
#                 plt.figure()
#                 plt.plot(rows['k'], rows[p])
#                 plt.xlabel('k')
#                 plt.ylabel(p)
#                 plt.title(f'{p} vs k')
#                 path = r'C:\Users\Cherry\Desktop\School\CSC311\Project\csc311\starter_code\part_b\plots' + f'\{l}_{lr}_{s}_{p}.png'
#                 plt.savefig(path)

print(df.loc[df['val_acc'] == df['val_acc'].max()])
print(df.loc[df['test_acc'] == df['test_acc'].max()])