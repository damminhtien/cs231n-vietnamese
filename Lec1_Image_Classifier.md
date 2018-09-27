# CS231n Mạng nơron tích chập cho nhận diện trực quan
## Intro to image classification, data-driven approach, pipeline (Giới thiệu bài toán phân loại ảnh, cách tiếp cận hướng dữ liệu, kĩ thuật đường ống)

**Từ khóa**: Phân loại ảnh, cách tiếp cận hướng dữ liệu, k-Nearest Neighbor, train/ val/ test splits
### Phân loại ảnh:
**Giới thiệu**: Phân loại ảnh là một trong những vấn đề quan trọng nhất của thị giác máy tính. Với một ảnh đầu vào, ta cần đưa ra ảnh này thuộc lớp nào (các lớp nằm trong một bộ cố định).

**Ví dụ**: Ảnh dưới đây cần đưa vào một trong bốn lớp (chó, mèo, mũ, cốc). Dưới góc độ của máy tính nhìn một bức ảnh, sẽ là một mảng ba chiều. Trong ví dụ, bức ảnh với 248 pixels chiều rộng, 400 pixels chiều cao và 3 màu RGB => Ma trận biểu diễn ảnh chứa 248*400*3 = 297600 số nguyên nằm trong khoảng từ 0 (đen) đến 255 (trắng).
  <img src="/assets/classify.png">

**Vấn đề**: có rất nhiều yếu tố ảnh hưởng đến một bức ảnh
  + Sự khác biệt về góc nhìn (viewpoint variation): cùng một đối tượng nhưng khác góc nhìn sẽ có các bức ảnh khác nhau
  + Sự thay đổi về kích thước (scale variation): một đối tượng có thể có nhiều kích thước do khoảng cách xa gần
  + Biến dạng (deformation): các đối tượng bị biến dạng
  + Sự che lấp (occulusion): đôi khi các đối tượng có kích thước rất nhỏ so với ảnh đầu vào
  + Điều kiện chiếu sáng (illumination conditions)
  + Bối cảnh lận cận: các đối tượng bị hòa lẫn vào nền của ảnh
  + Sự khác biệt trong nội tại (intra-class variation): cùng là một lớp đối tượng nhưng có nhiều nhiều loại khác nhau → do đó khác biệt về kích cỡ, kiểu dáng.
  <img src="/assets/challenges.jpeg">

Một mô hình tốt là mô hình ít bị ảnh hưởng bỏi những thông tin nhiễu.

### Cách tiếp cận hướng dữ liệu: 
<img src="/assets/trainset.jpg">
Với mỗi lớp đối tượng, thay vì phải viết thuật toán cụ thể để phân loại, ta sẽ đưa cho máy tính một tập dữ liệu để học, lần sau có các dữ liệu tương tự, máy tính có thể phân loại các dữ liệu này.

### Đường ống: 
  + Đầu vào: một tập N ảnh được gán bởi k nhãn → ta gọi là tập train.
  + Học: ta đưa tập train cho máy học. Quá trình này gọi là train a classifier hay learning a model.
  + Đánh giá: ta đánh giá một mô hình bằng 1 tập mà mô hình chưa nhìn thấy, gọi là tập test. Từ đó đưa ra tỷ lệ chính xác.

### Thuật toán láng giềng gần nhất (Nearest Neighbor):
Láng giềng gần nhất là thuật toán cơ bản và đơn giản trong các thuật toán học máy nên thường thì hiếm khi được dùng trong các bài toán phân loại ảnh vì độ chính xác thấp (thường sẽ dùng CNN) Tuy nhiên trong bài này, chúng ta sẽ thử tiếp cận với thuật toán này, để có thể lấy ý tưởng cơ bản nhất về các bài toán phân loại ảnh.
Về ý tưởng thì thuật toán láng giềng gần nhất là một thuật toán học có giám sát, với mỗi đối tượng, phân nó vào cùng lớp với láng giềng gần nó nhất. 

Ví dụ về tập dữ liệu: CIFAR-10 là tập dữ liệu mẫu nổi tiếng, bao gồm 60000 ảnh kích thước 32x32x3 chia thành 10 lớp. Trong 60000 ảnh này bao gồm 50000 ảnh tranning và 10000 ảnh test.
<img src="/assets/nn.jpg">
Nhiệm vụ của chúng ta là huấn luyện một mô hình có khả năng phân loại các ảnh này. Chúng ta sẽ dùng thuật toán láng giềng gần nhất để giải quyết.

Thuật toán láng giềng gần nhất đòi hỏi một định nghĩa về khoảng cách (như thế nào là gần? gần nhất?) Giả sử có 2 ảnh, 2 ảnh này là hai ma trận vector I1, I2, ta có thể tính ra khoảng cách giữa 2 ảnh này dựa trên L1 distance.
<img src="https://latex.codecogs.com/gif.latex?d_1&space;(I_1,&space;I_2)&space;=&space;\sum_{p}&space;\left|&space;I^p_1&space;-&space;I^p_2&space;\right|" title="d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right|" />
<img src="/assets/nneg.jpeg">

Lập trình:

```python
import numpy as np

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```
Test thử thuật toán đưa ra kết quả xấp xỉ 38.6% trên bộ CIFAR-10, mặc dù còn khá thấp so với đôi mắt của con người (xấp xỉ 94%) nhưng vẫn cao hơn nhiều so với việc ta dự đoán ngẫu nhiên (xác suất là 10% - do có 10 lớp). Điều này khiến ta nghĩ đến việc lựa chọn một cách tính khoảng cách tốt hơn nhằm tăng thêm độ chính xác.

**Lựa chọn cách tính khoảng cách**: có nhiều cách để tính khoảng cách giữa 2 vector, một trong những cách phổ biến khác đó là L2 distance:
<img src="https://latex.codecogs.com/gif.latex?d_2&space;(I_1,&space;I_2)&space;=&space;\sqrt{\sum_{p}&space;\left(&space;I^p_1&space;-&space;I^p_2&space;\right)^2}" title="d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}" />

Sửa lại code:
```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

Với L2, test lại với thuật toán láng giềng gần nhất, ta có kết quả mới xấp xỉ 35.4% thấp hơn so với L1… 

Vậy vậy lựa chọn khoảng cách ảnh hưởng khá nhiều tới kết quả của bài toán…

### k-Nearest neighbor classifier:
Có một cách khác để cải tiến giải thuật láng giềng gần nhất, đó là lựa chọn nhiều láng giềng hơn để 
tránh đi được các điểm nhiễu, các điểm ngoại lai.  Nhiều láng giềng hơn có thể giúp được quyết định phân loại đúng đắn hơn.
<img src="/assets/knn.jpeg">
Tuy vậy lại xuất hiện một vấn đề mới: chọn bao nhiêu k là hợp lí?

**Tập validation cho hyperparameter tuning**:

Chọn bao nhiêu k, hay chọn khoảng cách L1/L2 … có thể dẫn tới độ chính xác khác nhau. Việc có nhiều cách lựa chọn các tham số đầu vào cho một giải thuật, gọi là hyperparameter. Chọn parameters như thế nào là tốt nhất? Việc thay đổi các giá trị parameter sao cho mô hình đạt độ chính xác cao nhất, đó chính là hypermeter tuning.
Có một ý tưởng khá tốt là ta sẽ thử từng trường hợp, ghi lại và chọn ra trường hợp có kết quả cao nhất (grid search)? Không tồi. Tuy nhiên, với lượng tham số đầu vào lớn có thể làm mất khá nhiều thời gian, một giờ, thậm chí một ngày. 
Đặc biệt, chúng ta không được sử dụng tập test trong quá trình thử các tham số này. Việc dùng tập test để đánh giá khi hyperparameter tuning là không hề tốt chút nào khi thiết kế các giải thuật học máy. Đôi khi, với một tập tham số đầu vào này, kết quả cho ra rất tốt so với tập test, tuy nhiên khi đưa khi kiểm thử ở thực tế, kết quả lại tồi đi rất nhiều (overfitting). Có một nguyên lý là:

*Evaluate on the test set only a single time, at the very end.*

Vậy không có test set, làm sao ta có thể đánh giá mô hình để trong quá trình hyperparameter tuning.? May mắn thay, có một cách tốt hơn để giải quyết vấn đề này mà không phải động vào tập test…
Ý tưởng là chia tập train thành 2 phần, 1 dành cho train, 1 dành cho validation. Tập validation chính là có tác dụng như một tập test fake.

Với bài toán CIFAR-10, 50000 ảnh train có thể chia ra cho 49000 ảnh train và 1000 ảnh validation. Bây giờ, ta đã sẵn sàng cho phép thử các bộ parameter.

Lập trình:
```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

### Kiểm tra chéo (Cross validation):

Trong nhiều trường hợp, bộ trainning khá nhỏ, bắt buộc ta phải sử dụng một kĩ thuật tinh chỉnh hơn để hyperparameter tuning đó là cross validation.

Ví dụ có một tập train chia thành 5 phần bằng nhau lấy 1 phần ra làm validation → trainning → tính độ chính xác. Lặp lại 4 lần việc trên bằng cách lấy các phần khác nhau trong tập train. Khi có 5 kết quả, chia trung bình, đó là kết quả cuối cùng. 
<img src="/assets/crossval.jpeg">

Tuy vậy, trong thực tế, người ta thường tránh cross-validation vì khối lượng tính toán lớn. Thay vào đó, người ta thường chia tay luôn tập train ban đầu thành 10 – 50% cho tập validation. Số lượng parameter càng lớn thì chia cho tập validation càng nhiều ^^!
