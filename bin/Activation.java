package dl.utils;

import java.util.Arrays;
// import dl.loss.*;
import dl.optimizer.*;
// import dl.utils.*;
// import dl.layers.*;


public class Activation {
	private String name;
	private final String[] acitvation_funcs = new String[]{
		"ReLU",
		"Sigmoid",
		"Tanh",
		"Softmax"
	};

	public Activation(int index){
		if(index > acitvation_funcs.length || index < 0){
			System.out.println("[WARNING] activation function index not available ... ");
		}else{
			this.name = acitvation_funcs[index];
		}
	}

	public Activation(String name){
		boolean available = false;
		for(String avail_name : acitvation_funcs){
			if(name == avail_name){
				available =  true;
			}
		}

		if(available){
			this.name = name;
		}else{
			System.out.println("[WARNING] activation function not available ... ");
		}
	}

	public Vector forward(Vector input){
		double[] input_vec = input.getElements();
		double sum = 0;
		for (int i = 0; i < input_vec.length; i++){
			sum += input_vec[i];
		}

		for(int i = 0; i < input_vec.length; i++){
			switch(this.name){
				case "ReLU": // (x = x for x > 0) and (x = 0 for x < 0)
				{
					if(input_vec[i] < 0)
						input_vec[i] = 0;
					break;
				}
				case "Sigmoid": // x = 1/(1 + e^(-x))
				{
					input_vec[i] = 1/(1 + Math.pow(Math.exp(1), -input_vec[i]));
					break;
				}
				case "Tanh": // x = (e^2x - 1)/(e^2x + 1)
				{
					input_vec[i] = (Math.pow(Math.exp(1), 2*input_vec[i]) - 1)/(Math.pow(Math.exp(1), 2*input_vec[i]) + 1);
					break;
				}
				case "Softmax": // x[i] = x[i]/sum(X)
				{
					input_vec[i] = input_vec[i]/sum;
					break;
				}
			}
		} // end for loop

		return new Vector(input_vec, false);
	}
}