import scipy.io

def load_mat_data(path):
  """
  path - path to .mat file

  return: x - array of features
  return: y - array of labels
  """
  mat = scipy.io.loadmat(path)
  for k in mat.keys():
    if '__' not in k:
      data = mat[k][0][0]
      x = []
      for v in data[0].T:
        x.append(v.flatten().tolist())
      y = data[1].flatten().tolist()
      return x, y

if __name__ == '__main__':
  x_train, y_train = load_mat_data('train.mat')
  x_test, y_test = load_mat_data('test.mat')

  f = open('y_test', 'w')
  for v in y_test:
    f.write(chr(v))
  f.close()

  f = open('y_train', 'w')
  for v in y_train:
    f.write(chr(v))
  f.close()

  f = open('x_test', 'w')
  for v in x_test:
    for val in v:
      f.write(chr(val))
  f.close()

  f = open('x_train', 'w')
  for v in x_train:
    for val in v:
      f.write(chr(val))
  f.close()