package dl.loss;

import java.util.Arrays;
// import dl.loss.*;
import dl.optimizer.*;
import dl.utils.*;
// import dl.layers.*;

public enum Loss {
	// Entropy("Entropy"),
	MSE("Mean Squared Error");

	private String name;
	Loss(String name){
		this.name = name;
	}

	public static Matrix gradient(Matrix weights, Vector inputs, Vector outputs, Vector predictions){
		// this will return a matrix of gradient equals to the size of the weights matrix
		Matrix m = new Matrix(weights.getNumCols(), weights.getNumRows());

		Vector[] gradient_vectors = new Vector[weights.getNumCols()];
		for (int i = 0; i < weights.getNumCols(); ++i){
			// for each weight vector ...
			// gradient of w[i][j] = 2.x[j](y_hat[i] - y[i])
			Vector weight_vector = weights.getVector(i);
			double[] gradient_elems = new double[weight_vector.getElements().length];
			for (int j = 0; j < weight_vector.getElements().length; j++){
				double gradient = 0;
				gradient = 2 * inputs.getElements()[j] * (predictions.getElements()[i] - outputs.getElements()[i]);
				gradient_elems[j] = gradient;
			}

			Vector gradient_vector = new Vector(gradient_elems, true);
			gradient_vectors[i] = gradient_vector;
		}

		m = new Matrix(gradient_vectors);

		return m;
	}

	public double loss(Vector predictions, Vector targets){
		int len = predictions.getElements().length;
		double loss = 0;

		if(this.name == "Mean Squared Error"){
			for (int i = 0; i < len; i ++){
				loss += Math.pow((predictions.getElements()[i] - targets.getElements()[i]),2);
			}

			loss = loss / len;
		}

		return loss;
	}
}