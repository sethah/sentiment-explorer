import pytest

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from nbsvm import NBSVM


@pytest.fixture
def newsgroups():
    train = fetch_20newsgroups(
        subset="train",
        categories=['alt.atheism', 'sci.space']
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=['alt.atheism', 'sci.space']
    )
    vectorizer = CountVectorizer(binary=True)
    train_X = vectorizer.fit_transform(train.data)
    train_y = train.target
    test_X = vectorizer.transform(test.data)
    test_y = test.target
    return train_X, train_y, test_X, test_y


def test_NBSVM_initializes():
    clf = NBSVM()
    assert hasattr(clf, 'alpha')
    assert hasattr(clf, 'beta')
    assert hasattr(clf, 'C')


def test_NBSVM_initializes_with_params():
    clf = NBSVM(alpha=0.1, beta=0.2, C=0.3)
    assert hasattr(clf, 'alpha')
    assert hasattr(clf, 'beta')
    assert hasattr(clf, 'C')
    assert clf.alpha == 0.1
    assert clf.beta == 0.2
    assert clf.C == 0.3


def test_NBSVM_extracts_classes(newsgroups):
    X, y, _, _ = newsgroups
    clf = NBSVM()
    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert len(clf.classes_) == 2


def test_NBSVM_can_predict(newsgroups):
    X, y, _, _ = newsgroups
    clf = NBSVM()
    clf.fit(X, y)
    p = clf.predict(X)
    assert p.shape == y.shape


def test_NBSVM_scores_well_on_test(newsgroups):
    train_X, train_y, test_X, test_y = newsgroups
    clf = NBSVM()
    clf.fit(train_X, train_y)
    p = clf.predict(test_X)
    assert accuracy_score(p, test_y) > 0.9
