public class NetworkMathHandler {
    /*
    |------------------------------------------| NON-DERIVATIVE METHODS |------------------------------------------|
     */
    //DONE
    public static double feedForwardTest(double weightedSum){
        return weightedSum;
    }

    //DONE
    public static double TANH_Activation(double weightedSum){
        double lambda = 1; //Needed, not sure why :)

        double kk = weightedSum * lambda;
        double a = Math.exp(kk);
        double b = Math.exp(-kk);

        return  (a - b) / (a+b);
    }

    //TO DO
    public static double SIGMOID_Activation(double weightedSum){
        double lambda = 1;
        return 1.0 / (1.0 + Math.exp(-weightedSum));
    }


    /*
    |--------------------------------------------| DERIVATIVE METHODS |--------------------------------------------|
     */

    public static double TANH_Derivative(double weightedSum){
        double tanh = TANH_Activation(weightedSum);
        return 1 - (tanh * tanh);
    }

    public static double ReLU(double input){
        if(input <= 0){
            return 0;
        } else {
            return input;
        }
    }
    public static double ReLUDerivative(double input){
        if(input <= 0){
            return 0.01;
        } else {
            return 1;
        }
    }
}