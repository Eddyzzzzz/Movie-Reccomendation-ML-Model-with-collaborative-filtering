'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state

        # Initialize parameters with the correct dimensionality
        self.param_dict = dict(
            mu=ag_np.zeros(1),  # Global mean
            b_per_user=ag_np.zeros(n_users),  # User biases
            c_per_item=ag_np.zeros(n_items),  # Item biases
            U= 0.5 * random_state.randn(n_users, self.n_factors),  # User factors
            V= 0.5 * random_state.randn(n_items, self.n_factors),  # Item factors
        )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
        '''
        if mu is None:
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict['b_per_user']
        if c_per_item is None:
            c_per_item = self.param_dict['c_per_item']
        if U is None:
            U = self.param_dict['U']
        if V is None:
            V = self.param_dict['V']

        # Prediction: ŷij = μ + bi + cj + ui^T * vj
        yhat_N = (
            mu
            + b_per_user[user_id_N]
            + c_per_item[item_id_N]
            + ag_np.sum(U[user_id_N] * V[item_id_N], axis=1)
        )
        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        data_tuple : tuple
            Contains (user_ids, item_ids, ratings)

        Returns
        -------
        loss : float scalar
        '''
        y_N = data_tuple[2]  # True ratings
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        # Calculate mean squared error using ag_np.square
        data_loss = ag_np.mean(ag_np.square(y_N - yhat_N))

        # Calculate regularization loss using ag_np.square
        reg_loss = self.alpha * (
            ag_np.sum(ag_np.square(param_dict['U'])) + ag_np.sum(ag_np.square(param_dict['V']))
        )

        # Total loss
        loss_total = data_loss + reg_loss
        return loss_total


if __name__ == '__main__':
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

    # Create the model and initialize its parameters
    model = CollabFilterOneVectorPerItem(
        n_epochs=500, batch_size=100, step_size=0.2,
        n_factors=50, alpha=0.01)
    model.init_parameter_dict(n_users, n_items, train_tuple)
    print(f"U shape: {model.param_dict['U'].shape}")
    print(f"V shape: {model.param_dict['V'].shape}")    

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)

    # Plot training and validation MAE over epochs
    #final_train_mae = model.trace_mae_train[-1]
    final_valid_mae = model.trace_mae_valid[-1]

    #print(f"Final Training MAE: {final_train_mae:.4f}")
    print(f"Final Validation MAE: {final_valid_mae:.4f}")

    user_ids, item_ids, ratings = test_tuple  # Unpack test data
    predicted_ratings = model.predict(user_ids, item_ids)  # Get predictions
    mae = ag_np.mean(ag_np.abs(predicted_ratings - ratings))
    print(f"Test Set MAE: {mae:.4f}")

    import matplotlib.pyplot as plt
    #plt.plot(model.trace_epoch, model.trace_mae_train,'.-',label='Train MAE')
    plt.plot(model.trace_epoch, model.trace_mae_valid,'.-',label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

    