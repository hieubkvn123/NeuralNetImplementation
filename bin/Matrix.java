package dl.utils;

import java.util.Arrays;
// import dl.loss.*;
import dl.optimizer.*;
// import dl.utils.*;
// import dl.layers.*;

public class Matrix {
	private Vector[] element_vectors; // maximum 1024 vectors in a matrix
	private int numCols = 0;
	private int numRows = 0;

	public Matrix(int numCols, int numRows){
		this.numCols = numCols;
		this.numRows = numRows;

		// initialize the vector with arrays of ones
		double[] elems_in_vec = new double[numRows];
		Arrays.fill(elems_in_vec, 1);

		Vector vector_in_mat = new Vector(elems_in_vec, true); // these vectors are initially transposed
		this.element_vectors = new Vector[numCols];
		Arrays.fill(this.element_vectors, vector_in_mat);
	}

	public int getNumCols(){
		return this.numCols;
	}

	public int getNumRows(){
		return this.numRows;
	}
	public Matrix(Vector[] element_vectors){
		this.element_vectors = element_vectors;
		this.numCols = element_vectors.length;
		this.numRows = element_vectors[0].getElements().length;
	} 

	public Vector getVector(int row_index){
		return this.element_vectors[row_index];
	}

	@Override
	public String toString(){
		String str = "[";
		for (int i = 0; i < this.element_vectors.length; i++){
			if(i == this.element_vectors.length - 1){ // if the last vector
				str = str + " " + this.element_vectors[i].toString();
			}

			else if(i == 0){ // if the first vector
				str = str + this.element_vectors[i].toString() + "\n";
			}

			else{ // other cases
				str = str + " " + this.element_vectors[i].toString() + "\n";
			}
		}

		str = str + "]";
		return str;
	}
}