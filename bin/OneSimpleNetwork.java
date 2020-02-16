import java.util.Arrays;
import dl.loss.Loss;
import dl.optimizer.Optimizer;
import dl.utils.Vector;
import dl.utils.Matrix;
import dl.layers.Layer;

public class OneSimpleNetwork {
	public static void main(String[] args){
		double[] input_elements = new double[]{2,4,1,5,3}; // 5 elements
		double[] output_elements = new double[]{1,0}; // 2 elements 
		int epochs = 20;
		int i = 0;

		Vector input = new Vector(input_elements, false);
		Vector output = new Vector(output_elements, false);

		Dense net = new Dense(input, output); // a model consisting of one single layer

		while(!net.getConverged()){
			i++;
			net.forward();
			net.backward();

			System.out.println("_________________________________________________________");
			System.out.println("After epoch number : " + (net.getEpochs()+1));

			System.out.println(net);
		}

		System.out.println("Number of epochs after converged : " + net.getEpochs());
		//System.out.println(net);
	}
}