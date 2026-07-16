import os
import numpy as np
import matplotlib.pyplot as plt

#eisagogi vivliothikon apo sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report

def load_imdb_data(data_dir):
    #fortosi tou dataset apo ton fakelo

    def load_split(split_dir):
        """Fortosi mias ypomonadas (train i test)."""
        texts, labels = [], []

        for label_type in ['pos', 'neg']:
            dir_path = os.path.join(split_dir, label_type)
            label = 1 if label_type == 'pos' else 0  # positive=1, negative=0

            #diavazei ola ta arxeia kritikon ston fakelo
            for filename in os.listdir(dir_path):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)

        return texts, labels

    #fortosi train kai test set
    train_texts, train_labels = load_split(os.path.join(data_dir, 'train'))
    test_texts, test_labels = load_split(os.path.join(data_dir, 'test'))

    return (train_texts, train_labels), (test_texts, test_labels)



#aksiologisi kai diagrammata
def compute_metrics(y_true, y_pred):
# ypologismos precision, recall, F1 gia kathe klasi kai meso oro
#epistrefei dictionary me metrikes ana klasi kai average

    #metrikes gia kathe klasi xexorista (negative=0, positive=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None
    )
    #micro kai macro averages pou tha prepei nane idia
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    return {
        'negative': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0]},
        'positive': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1]},
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f1},
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1}
    }


def print_metrics_table(metrics, title="Results"):
    """Ektyposi pinaka me ola ta metrikes apotelesmatwn."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*60}")

    for class_name in ['negative', 'positive', 'macro', 'micro']:
        m = metrics[class_name]
        print(f"{class_name:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")

    print(f"{'='*60}\n")

#dimiourgia sinartiseon
def plot_learning_curves(train_sizes, train_metrics, dev_metrics, metric_name, title_prefix, save_path):
    train_scores = [m[metric_name] for m in train_metrics]
    dev_scores = [m[metric_name] for m in dev_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label=f'Training', color='blue')
    plt.plot(train_sizes, dev_scores, 'o-', label=f'Development', color='orange')

    plt.xlabel('Number of Training Examples')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{title_prefix} - {metric_name.capitalize()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

#main
def main():
    print("-"*70)
    print("IMDB analisi tainion xrisimopiontas afelhs taksinomisi Bayes me polimetavliti morfi Bernoulli apo sklearn")
    #uperparametroi
    #afairesi ton 50 pio sixnon lekseon
    N_MOST_FREQUENT = 50
    #afairesi lexeon pou emfanizontai se <100 keimena
    K_MOST_RARE = 100
    #teliko megethos lexilogiou
    M_VOCAB_SIZE = 2000
    #laplace gia ton afelh Bayes
    ALPHA = 0.5
    #pososto dedomenwn gia to development set 10%
    DEV_RATIO = 0.1
    #fakelos dataset
    DATA_DIR = './aclImdb'
    print('='*70)
    print("Fortosi Dedomenwn:")
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_data(DATA_DIR)

    print(f" Fortwthikan {len(train_texts)} training, {len(test_texts)} test keimena")

    #stratify simenei oti kratame tin idia analogia positive/negative oste na min uparxei anakrivia kai sto train kai sto dev
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        train_texts, train_labels,
        test_size=DEV_RATIO,
        random_state=42,
        stratify=train_labels
    )

    print(f"  Diaxorismos: {len(train_texts)} train, {len(dev_texts)} dev, {len(test_texts)} test")

    print('='*70)
    print("Dianysmatopoiisi keimenon me CountVectorizer:")
    #vriskoume posa keimena periexei kathe lexi gia na afairesoume tis pio syxnes kai tis pio spanies
    temp_vectorizer = CountVectorizer(binary=True)
    temp_X = temp_vectorizer.fit_transform(train_texts)

    #se posa keimena emfanizetai kathe lexi
    doc_frequencies = np.asarray(temp_X.sum(axis=0)).flatten()
    vocabulary = temp_vectorizer.get_feature_names_out()

    print(f"  Synolikes monadikes leksis prin to filtrаrisma: {len(vocabulary)}")

    # Taksinomisi lexeon kata ftinousa sychnotita gia na aferesoume
    sorted_indices = np.argsort(doc_frequencies)[::-1]

    #aferesi pio sixnon
    sorted_indices = sorted_indices[N_MOST_FREQUENT:]
    #aferesi pio spanion
    valid_indices = sorted_indices[doc_frequencies[sorted_indices] >= K_MOST_RARE]
    #kratisi mono ton egkyron lexeon
    valid_vocabulary = vocabulary[valid_indices]

    n_removed_frequent = N_MOST_FREQUENT
    n_removed_rare = len(vocabulary) - N_MOST_FREQUENT - len(valid_vocabulary)
    print(f"  Aferethikan: {n_removed_frequent} sixnes + {n_removed_rare} spanies = {n_removed_frequent + n_removed_rare} synolika")
    print(f"  Parameinan: {len(valid_vocabulary)} leksis")

    #binary=True giati kanoume polionimiki ilopiisi(dld yparxi i den yparxi i lexi)
    vectorizer = CountVectorizer(binary=True, vocabulary=valid_vocabulary)

    X_train_full = vectorizer.fit_transform(train_texts)
    X_dev_full = vectorizer.transform(dev_texts)
    X_test_full = vectorizer.transform(test_texts)

    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)

    print('='*70)
    print(f"Epilogi ton kalyteron {M_VOCAB_SIZE} xarakthristikon:")

    #epilogi ton M kalyteron lexeon
    #mutual information = information gain
    #metraei poso pliroforisi prosferei mia lexi gia tin taksinomisi
    selector = SelectKBest(mutual_info_classif, k=M_VOCAB_SIZE)
    X_train = selector.fit_transform(X_train_full, y_train)
    X_dev = selector.transform(X_dev_full)
    X_test = selector.transform(X_test_full)

    print(f"  Teliko megethos pinaka xarakthristikon: {X_train.shape}")

    #anaktisi onomaton epilegmenon lexeon
    feature_names = np.array(vectorizer.get_feature_names_out())
    selected_mask = selector.get_support()
    selected_features = feature_names[selected_mask]

    #emfanisi ton 10 kalyterοn xaraktiristikon vasei MI score
    mi_scores = selector.scores_[selected_mask]
    top_indices = np.argsort(mi_scores)[::-1][:10]
    print(f"  Top 10 lexeis me to megalytero MI: {list(selected_features[top_indices])}")

    print("\n" + "="*70)
    print("Dimiourgia Kampilon Mathisis:")

    #upologismos diaforon megethon training set (apo 10% eos 100%)
    n_train = X_train.shape[0]
    train_sizes = np.linspace(0.1, 1.0, 10)  # 10 diaforetika megethi
    train_sizes = (train_sizes * n_train).astype(int)

    train_metrics_list = []
    dev_metrics_list = []

    #tyxaia anakatassi gia na paroume diaforetika ypοsynola
    np.random.seed(42)
    indices = np.random.permutation(n_train)

    #gia kathe megethos training set
    for size in train_sizes:
        #epilogi ton proton 'size' paradeigmaton
        idx = indices[:size]
        X_sub = X_train[idx]
        y_sub = y_train[idx]

        #ekpaideysi BernoulliNB apo sklearn
        clf = BernoulliNB(alpha=ALPHA)
        clf.fit(X_sub, y_sub)

        #aksiologisi sto training subset (mono gia positive class)
        y_train_pred = clf.predict(X_sub)
        p, r, f1, _ = precision_recall_fscore_support(y_sub, y_train_pred, labels=[1], average=None)
        train_metrics_list.append({'precision': p[0], 'recall': r[0], 'f1': f1[0]})

        #aksiologisi sto development set (mono gia positive class)
        y_dev_pred = clf.predict(X_dev)
        p, r, f1, _ = precision_recall_fscore_support(y_dev, y_dev_pred, labels=[1], average=None)
        dev_metrics_list.append({'precision': p[0], 'recall': r[0], 'f1': f1[0]})

        print(f"  Train size: {size:5d} | Train F1: {train_metrics_list[-1]['f1']:.4f} | Dev F1: {dev_metrics_list[-1]['f1']:.4f}")

    #apothikeusi diagrammaton gia precision, recall, F1
    for metric in ['precision', 'recall', 'f1']:
        plot_learning_curves(
            train_sizes, train_metrics_list, dev_metrics_list,
            metric, 'Bernoulli Naive Bayes', f'learning_curve_{metric}.png'
        )
    
    print("\n" + "="*70)
    print("Teliki Axiologisi sto Test Set")

    #ekpaideysi sto synoliko training set
    clf = BernoulliNB(alpha=ALPHA)
    clf.fit(X_train, y_train)

    #provlepsi sto test set
    y_test_pred = clf.predict(X_test)

    #ypologismos kai ektyposi metrikon
    metrics = compute_metrics(y_test, y_test_pred)
    print_metrics_table(metrics, "Apotelesmata sto Test Set - Bernoulli Naive Bayes")

    print("\nOloklirosi!")


if __name__ == "__main__":
    main()
