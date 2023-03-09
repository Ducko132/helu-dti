
class EarlyStopper():
  """
  Implements the early stop mechanism
  """

  def __init__(self, tolerance=5, min_delta=0):

    self.tolerance = tolerance
    self.min_delta = min_delta
    self.counter = 0

  def early_stop(self, train_loss, validation_loss):
    if (validation_loss - train_loss) > self.min_delta:
      self.counter += 1
      if self.counter >= self.tolerance:
        return True
    return False
