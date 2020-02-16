package dl.utils;

import java.util.Arrays;
// import dl.loss.*;
import dl.optimizer.*;
// import dl.utils.*;
// import dl.layers.*;

public class Vector {
	private double[] elements = new double[1024]; // max number of elements = 1024
	private boolean transpose = false; // by default all vectors are column vectors

	public Vector(double[] elements, boolean transpose){
		this.elements = elements;
		this.transpose = transpose;
	}

	public void setElements(double[] elements){
		if(elements.length > 1024){
			System.out.println("[ERROR] Index out of bound, please reset element ... ");
		}else{
			this.elements = elements;
		}
	} 

	public double[] getElements(){
		return this.elements;
	}

	public void setTranspose(boolean transpose){
		this.transpose = transpose;
	}

	public boolean getTranspose(){
		return this.transpose;
	}

	public boolean isScalar(){
		boolean scalar = false;
		if(this.getElements().length == 1){
			scalar = true;
		}

		return scalar;
	}

	public double dot(Vector vector){
		double dot_product = 0;
		if(this.elements.length == vector.getElements().length){
			for(int i = 0; i < this.elements.length; i++){
				dot_product += (this.elements[i] * vector.getElements()[i]);
			}
		}else{
			System.out.println("[ERROR] Vectors length not matched ...");
		}

		return dot_product;
	}

	public double sum(){
		double sum = 0;
		for (int i = 0; i < this.elements.length; i++){
			sum += this.elements[i];
		}

		return sum;
	}

	public double euclidean_dist(Vector vector){
		double element_wise_sum_square = 0;
		for (int i = 0; i < vector.getElements().length; i++){
			element_wise_sum_square += Math.pow((this.getElements()[i] - vector.getElements()[i]),2);
		}

		return Math.sqrt(element_wise_sum_square);
	}

	@Override
	public String toString(){
		String str = "[ ";
		for (int i = 0; i < this.elements.length; i++){
			str = str + String.format("%.2f",this.elements[i]) + " ";
		}

		str = str + "]";

		if(this.transpose){
			str = str + "T";
		}

		return str;
	}
}
