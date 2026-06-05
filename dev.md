Accuracy: доля правильных предсказаний (pred_classes == true_classes), быстрый общий показатель.

Precision / Recall: точность и полнота по положительному классу (TP/(TP+FP), TP/(TP+FN)); важно при разной стоимости ошибок.

F1-score: гармоническое среднее precision и recall — хорошая сводная метрика при несбалансированных классах. F1 = 2 * (Precision * Recall) / (Precision + Recall)
 F1 показывает одновременно насколько хорошо модель находит объекты положительного класса из всех объектов положительного класса и какая доля из тех, кого алгоритм назвал положительным классом, действительно являются положительным классом.


============================================================
 а почему stochastic а не linear ?

Коротко — потому что «stochastic» (мини‑батчи) даёт практические преимущества по скорости и памяти по сравнению с полным (batch/linear) градиентным спуском.
Что значит:
Batch (полный, «linear»): на каждом шаге считаем градиент по всему набору и обновляем параметры один раз за эпоху.
Stochastic / Mini‑batch: обновляем параметры чаще — после каждого примера (стохастический) или после каждого мини‑батча (практический случай).
Почему обычно выбирают stochastic / mini‑batch:

Быстрее итерации: можно делать много небольших обновлений без прохода по всему датасету.
Меньше памяти: не нужно хранить/пропускать весь датасет за раз.
Лучшая общая сходимость в практике: шум в обновлениях помогает выходить из плоских минимумов и часто даёт лучшую генерализацию.
Удобно для онлайн/потоковых данных и для распараллеливания на GPU/батчах.
Современные адаптивные оптимизаторы (RMSProp, Adam) спроектированы под мини‑батчи.
Минусы stochastic:

Шумные обновления; требуется подобрать скорость обучения и размер батча.
Меньше точности на каждой итерации (но обычно компенсируется количеством итераций).
Практический компромисс:

Mini‑batch часто берут 32–256; ваш код уже делает mini‑batch SGD (параметр -b).
Если хотите полную (batch) версию — выставьте -b равным размеру тренировочного набора (или len(X_train)), тогда обновление будет по всему датасету за шаг.



=====================================================================
1. Feedforward (propagation avant)
Le feedforward est le passage des données à travers le réseau, de l'entrée vers la sortie.

Entrée
Couche cachée
Couche de sortie
Prédiction

2. Gradient Descent (descente de gradient)
Après la prédiction, on calcule l'erreur entre la sortie du modèle et la vraie réponse.
L'objectif est de minimiser cette erreur.
La descente de gradient modifie progressivement les poids du réseau :

w = w − η ∂L/∂w 
w est un poids ;
L est la fonction de perte (loss) ;
η est le taux d'apprentissage (learning rate).

On peut imaginer une personne qui descend une montagne en suivant la pente la plus forte vers le bas.

3. Backpropagation (rétropropagation)

La rétropropagation sert à calculer l'influence de chaque poids sur l'erreur finale.

Le calcul se fait en sens inverse :

Sortie
  ↑
Couche cachée
  ↑
Entrée

Grâce à la règle de la chaîne (chain rule), le réseau calcule les dérivées de l'erreur par rapport à chaque poids.

Ces dérivées (gradients) indiquent :

Quel poids doit être modifié ?
De combien ?
Dans quelle direction ?

Cycle complet d'entraînement
1. Feedforward
   ↓
2. Calcul de la perte (loss)
   ↓
3. Backpropagation
   ↓
4. Gradient Descent
   ↓
5. Mise à jour des poids
   ↓
Répéter

En résumé :

Feedforward → produire une prédiction.
Backpropagation → calculer les gradients.
Gradient Descent → mettre à jour les poids pour réduire l'erreur.

====================================================

L'overfitting (ou surapprentissage) se produit lorsqu'un modèle apprend trop bien les données d'entraînement, y compris le bruit et les particularités accidentelles, au lieu d'apprendre les règles générales.
Résultat :

excellente performance sur les données d'entraînement ;
mauvaise performance sur de nouvelles données jamais vues.