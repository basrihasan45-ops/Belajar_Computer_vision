import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

lfw_dataset = fetch_lfw_people(min_faces_per_person=300)
_, h, w=lfw_dataset.images.shape
X = lfw_dataset.data
y = lfw_dataset.target
target_names=lfw_dataset.target_names
def load_image_file(path):
    try:
        # Buka gambar
        img = Image.open(path)
        
        # Pastikan format RGB
        img = img.convert('RGB')
        
        # Ubah ke numpy array
        img_array = np.array(img)
        
        return img_array
    
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca gambar: {e}")
        return None
    

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

pca=PCA(n_components=100, whiten=True).fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

clf=MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

y_pred=clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

def flot_pict(images, titles, h, w, rows=3, cols=4):
    plt.figure(figsize=(16,10), dpi=80)
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[1])
        plt.xticks(())
        plt.yticks(()) 
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name=target_names[y_pred[i]].split(' ')[-1]
        tru_name=target_names[y_test[i]].split(' ')[-1]
        yield 'predicted:{0}\ntrue:{1}'.format(pred_name, tru_name)
prediction_title=list(titles(y_pred, y_test, target_names))
flot_pict(X_test, prediction_title, h, w)