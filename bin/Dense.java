import java.util.Arrays;
import dl.loss.Loss;
import dl.optimizer.Optimizer;
import dl.utils.Vector;
import dl.utils.Matrix;
import dl.layers.Layer;

public class Dense extends Layer {
	//private Vector input;
	//private Vector output;
	//private Vector predictions;
	//private Matrix weights;
	//private Matrix gradients;

	private Dense next;
	private double learning_rate;
	private Optimizer optimizer = Optimizer.SGD; // the default optimizer
	private Loss loss = Loss.MSE;

	private int epochs = 0;
	private boolean converged = false;

	public Dense(Vector input, Vector output){
		//this.input = input;
		//this.output = output;
		//this.weights = new Matrix(this.output.getElements().length, this.input.getElements().length);

		//double[] prediction_elems = new double[this.output.getElements().length];
		//for (int i = 0; i < this.output.getElements().length; i++){
		//	prediction_elems[i] = this.input.dot(this.weights.getVector(i));
		//}

		// this will initialize input, output, weights matrix and prediction vector
		super(input, output);
		this.learning_rate = 0.01;

	}

	// We need a constructor that takes in an input Vector only
	// and initialize the weight as all one
	// the forward the layer once
	// set the output as the prediction.
	//public Dense(Vector input){
		// the output is the next layer's input
		// super(input, this.next.getInput());
		// this.learning_rate = 0.01;
	//}

	public int getEpochs(){
		return this.epochs;
	}

	public boolean getConverged(){
		return this.converged;
	}

	public void forward(){
		this.epochs += 1;
		double[] prediction_elems = new double[this.getOutput().getElements().length];
		for (int i = 0; i < this.getOutput().getElements().length; i++){
			prediction_elems[i] = this.getInput().dot(this.getWeights().getVector(i));
		}

		this.setPrediction(new Vector(prediction_elems, false));
	}

	public void backward(){
		// the update rule for SGD : [w[i][j]](t+1) = [w[i][j]](t) - lr * g[i][j]
		Vector[] updated_vectors = new Vector[this.getWeights().getNumCols()];


		this.setGradients(Loss.gradient(this.getWeights(), this.getInput(), this.getOutput(), this.getPrediction()));
		double loss = this.loss.loss(this.getPrediction(), this.getOutput()); 

		if(loss < 1e-10){
			this.converged = true;
		}

		for (int i = 0; i < this.getWeights().getNumCols(); ++i){
			Vector weight_v = this.getWeights().getVector(i);
			Vector gradient_v = this.getGradients().getVector(i);

			int len = weight_v.getElements().length;
			double[] updated = new double[len];

			if(this.optimizer == Optimizer.SGD){
				for (int j = 0; j < len; ++j){
					double w = weight_v.getElements()[j];
					double g = gradient_v.getElements()[j];
					w = optimizer.update(w, g, this.learning_rate, this.epochs);

					updated[j] = w;
				}
			}else if(this.optimizer == Optimizer.ADAM){
				for (int j = 0; j < len; ++j){
					double w = weight_v.getElements()[j];
					double g = gradient_v.getElements()[j];
					w = optimizer.update(w, g, this.learning_rate, this.epochs);

					updated[j] = w;
				}
			}

			Vector updated_vec = new Vector(updated, true);
			updated_vectors[i] = updated_vec;
		}
		// update the weights
		this.setWeights(new Matrix(updated_vectors));
		// update the gradients
	}

	@Override
	public String toString(){
		String str = "";
		String adjusted = "";

		if(this.epochs == 0){
			adjusted = "Initial";
		}else{
			adjusted = "Adjusted";
		}

		str += "Input (" + adjusted + ") : \n";
		str += this.getInput().toString() + "\n\n";

		str += "Output (" + adjusted + ") : \n";
		str += this.getOutput().toString() + "\n\n";
		
		str += "Predictions (" + adjusted + ") : \n";
		str += this.getPrediction().toString() + "\n\n"; 

		str += "Weights (" + adjusted + ") : \n";
		str += this.getWeights().toString() + "\n\n";

		str += "Gradients (" + adjusted + ") : \n";
		str += this.getGradients().toString() + "\n\n"; 

		return str;
	}
}
