import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from libs.history import load_histories, plot_loss_curve
from histories import plot_train_recall_curve, plot_val_recall_curve

lhist = load_histories('../histories/k-fold-50')
plt_loss = plot_loss_curve(lhist=lhist, ep = 30)
plt_train_recall = plot_train_recall_curve(lhist=lhist, ep = 30)
plt_val_recall = plot_val_recall_curve(lhist=lhist, ep = 30)


plt_loss.show()
plt_train_recall.show()
plt_val_recall.show()
