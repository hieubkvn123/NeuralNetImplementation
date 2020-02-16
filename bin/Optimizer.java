package dl.optimizer;

import java.util.Arrays;
import dl.loss.*;
// import dl.optimizer.*;
import dl.utils.*;
import dl.layers.*;

public enum Optimizer {
	SGD("Stochastic Gradient Descent"),
	ADAM("Adaptive Moment Estimation");


	private double momentum_1 = 0;
	private double momentum_2 = 0;

	private double beta_1 = 0.9;
	private double beta_2 = 0.999;
	private double e = 10e-8;

	private String name;
	Optimizer(String name){
		this.name = name;
	}

	public double update(double w, double g, double lr, int i){
		// w = weight, g = gradient
		double updated = 0;

		if (this.name == "Stochastic Gradient Descent"){
			updated = w - lr * g;
		}if(this.name == "Adaptive Moment Estimation"){
			this.momentum_1 = this.beta_1 * this.momentum_1 + (1 - this.beta_1) * g;
			this.momentum_2 = this.beta_2 * this.momentum_2 + (1 - this.beta_2) * Math.pow(g,2);

			double corrected_momentum_1 = this.momentum_1 / (1 - Math.pow(this.beta_1, i));
			double corrected_momentum_2 = this.momentum_2 / (1 - Math.pow(this.beta_2, i));

			// System.out.println("Momentum 1 on epoch " + i + ": " + this.momentum_1);
			// System.out.println("Momentum 2 on epoch " + i + ": " + this.momentum_2);

			updated = w - (lr * (corrected_momentum_1 / (Math.sqrt(corrected_momentum_2) + this.e)));
		}

		return updated;
	}
}
