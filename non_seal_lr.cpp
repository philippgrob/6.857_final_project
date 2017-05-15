#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath> 

using namespace std;

void print_example_banner(string title);
void example_lr();

class Matrix {
   public:
      Matrix add_matrix(Matrix & other) {
         cout << "Adding two matricies together..." << endl;
         std::vector< std::vector<double> > result;
         for (int i = 0; i < n_rows; ++i)
         {
            std::vector<double> row_result;
            for (int j= 0; j < n_cols; ++j){
               double new_value =  values[i][j]+ other.values[i][j];
               row_result.emplace_back(new_value);
            }
            result.emplace_back(row_result);
         }
         return Matrix(result);
      }
        
      Matrix multiply(Matrix & other) {
         cout << "Multiplying 2 matricies together..."  <<endl;
         int rows = n_rows; 
         int cols = other.n_cols;
         std::vector< std::vector<double> > result = initialize_empty(rows, cols);
         for (int i = 0; i < n_rows; ++i)
         {
            for (int j= 0; j < other.n_cols; ++j){
                for (int k=0; k < n_cols; ++k){
                    double product = values[i][k] * other.values[k][j];
                    double new_value = product + result[i][j];
                    
                    result[i][j] = new_value;
                }
            }
         }
         return Matrix(result);
      }

      Matrix multiply_constant(double constant) {
         cout << "Mutliplying a matrix by a constant..." <<endl;
         std::vector< std::vector<double> > result;
         for (int i = 0; i < n_rows; ++i)
         {
            std::vector<double> row_result;
            for (int j= 0; j < n_cols; ++j){
               double new_value = values[i][j]* constant;
               row_result.emplace_back(new_value);
            }
           result.emplace_back(row_result);
         }
         return Matrix(result);
      }

      Matrix get_transpose(){
         cout << "Calculating the transpose of a matrix..." <<endl;
         std::vector< std::vector<double> > result;
         for (int i = 0; i < n_cols; ++i)
         {
            std::vector<double> row_result;
            for (int j= 0; j < n_rows; ++j){
               double new_value = values[j][i];
               row_result.emplace_back(new_value);
            }
           result.emplace_back(row_result);
         }
         return Matrix(result);
      }

      Matrix get_adjugate(){
        cout << "Calculating the adjugate of a matrix..." <<endl;
        if (n_cols == 1 && n_rows == 1){
           std::vector<double> row = {1};
           std::vector< std::vector<double> > result = {row};
           return Matrix(result);
        }
        if (n_cols == 2 && n_rows == 2){
            std::vector< std::vector<double> > result = initialize_empty(2, 2);
            //for 2x2 matrix [[a,b],[c,d]] adjugate is [[d, -b],[-c, a]]
            result[0][0] = values[1][1];
            result[0][1] = -1* values[0][1];
            result[1][0] = -1* values[1][0];
            result[1][1] = values[0][0];
            return Matrix(result);
        }
        else if (n_cols == 3 && n_rows == 3){
            std::vector< std::vector<double>> m_0_0_values = {{values[1][1], values[1][2]}, {values[2][1], values[2][2]}};
            Matrix m_0_0 =Matrix(m_0_0_values);
            std::vector< std::vector<double>> m_0_1_values = {{values[0][1], values[0][2]}, {values[2][1], values[2][2]}};
            Matrix m_0_1 =Matrix(m_0_1_values);
            std::vector< std::vector<double>> m_0_2_values = {{values[0][1], values[0][2]}, {values[1][1], values[1][2]}};
            Matrix m_0_2 =Matrix(m_0_2_values);
            double a_0_0 = m_0_0.get_determinant();
            double a_0_1 = -1 * m_0_1.get_determinant();
            double a_0_2 = m_0_2.get_determinant();


            std::vector< std::vector<double>> m_1_0_values = {{values[1][0], values[1][2]}, {values[2][0], values[2][2]}};
            Matrix m_1_0 =Matrix(m_1_0_values);
            std::vector< std::vector<double>> m_1_1_values = {{values[0][0], values[0][2]}, {values[2][0], values[2][2]}};
            Matrix m_1_1 =Matrix(m_1_1_values);
            std::vector< std::vector<double>> m_1_2_values = {{values[0][0], values[0][2]}, {values[1][0], values[1][2]}};
            Matrix m_1_2 =Matrix(m_1_2_values);

            double a_1_0 = -1*m_1_0.get_determinant();
            double a_1_1 = m_1_1.get_determinant();
            double a_1_2 = -1*m_1_2.get_determinant();

            std::vector< std::vector<double>> m_2_0_values = {{values[1][0], values[1][1]}, {values[2][0], values[2][1]}};
            Matrix m_2_0 =Matrix(m_2_0_values);
            std::vector< std::vector<double>> m_2_1_values = {{values[0][0], values[0][1]}, {values[2][0], values[2][1]}};
            Matrix m_2_1 =Matrix(m_2_1_values);
            std::vector< std::vector<double>> m_2_2_values = {{values[0][0], values[0][1]}, {values[1][0], values[1][1]}};
            Matrix m_2_2 =Matrix(m_2_2_values);

            double a_2_0 = m_2_0.get_determinant();
            double a_2_1 = -1*m_2_1.get_determinant();
            double a_2_2 = m_2_2.get_determinant();
            
            std::vector< std::vector<double>> result = {{a_0_0, a_0_1, a_0_2}, {a_1_0, a_1_1, a_1_2}, {a_2_0, a_2_1, a_2_2}};

            return Matrix(result);
        }
        else{
            return get_adjugate_recursive();
        }

      }
      Matrix get_adjugate_recursive(){
        cout << "Calculating the adjugate of a matrix..." <<endl;
        std::vector< std::vector<double> > result;
        for(int i=0; i < n_rows; ++i){
          std::vector<double> row_result;
          for(int j=0; j < n_cols; ++j){
            Matrix m = wo_row_col(i, j);
            int sign = std::pow(-1,i+j);
            double det = m.get_determinant();
            double value = det *sign;
            row_result.emplace_back(value);
          }
          result.emplace_back(row_result);
        }
        Matrix adj_T = Matrix(result);
        return adj_T.get_transpose();
      }

      Matrix wo_row_col(int row, int col){
        std::vector< std::vector<double> > result;
        for(int i=0; i < n_rows; ++i){
          if(i != row){
            std::vector<double> row_result;
            for(int j=0; j < n_cols; ++j){
              if(j != col){
                row_result.emplace_back(values[i][j]);
              }
            }
            result.emplace_back(row_result);
          }
        }
        return Matrix(result);
      }


      void print_decrypted(){
         cout << "Printing matrix..." <<endl;
         for (int i = 0; i < n_rows; ++i)
         {
            for (int j= 0; j < n_cols; ++j){
               cout << values[i][j];
               cout << "\t";
            }
            cout << endl;
         }
      }
 
    double get_determinant(){
        if (n_cols == 1 && n_rows == 1){
            return values[0][0];
        }
        if (n_cols == 2 && n_rows == 2){
            cout << "calculating determinat of a 2x2 matrix" <<endl;
            double ac = values[0][0] * values[1][1];
            double bd = values[0][1] * values[1][0];
            double det = ac- bd;
            return det;
        }
        else if (n_cols == 3 && n_rows == 3){
            cout << "calculating determinat of a 3x3 matrix" <<endl;
            double a = values[0][0];
            double b = values[0][1];
            double c = values[0][2];
            double d = values[1][0];
            double e = values[1][1];
            double f = values[1][2];
            double g = values[2][0];
            double h = values[2][1];
            double i = values[2][2];
            std::vector<double> aei_vec = {a, e, i};
            std::vector<double> bfg_vec = {b, f, g};
            std::vector<double> cdh_vec = {c, d, h};
            std::vector<double> ceg_vec = {c, e, g};
            std::vector<double> bdi_vec = {b, d, i};
            std::vector<double> afh_vec = {a, f, h};
            double aei = a * e * i;
            double bfg = b*f*g;
            double cdh = c*d*h;
            double ceg = -1*c*e*g;
            double bdi = -1*b*d*i;
            double afh = -1* a*f*h;
            double det = aei+ bfg+ cdh+ ceg+ bdi+ afh;
            return det;
        }
        else{
          return get_determinant_recursive();
        }
    }
    double get_determinant_recursive(){
        if(n_cols != n_rows){
          cout << "NOT A SQUARE MATRIX NO DETERMINANT";
          cout << n_cols << "\t" << n_rows << endl;
        }
        std::vector< std::vector<double> > products;
        int i=1;
        for(int j=0; j < n_cols; ++j){
          cout << "inside loop" << endl;
          Matrix m = wo_row_col(i, j);
          cout << "created matrix without row/col" << endl;
          double sign = std::pow(-1,i+j);
          cout << "right before recursive call" << endl;
          double det = m.get_determinant();
          cout << "finished recursive call" << endl;
          std::vector<double> to_mul = {values[i][j], det, sign};
          cout << "finished vector" << endl;
          products.emplace_back(to_mul);
          cout << "end of for loop" << endl;
        }
        std::vector<double> to_sum;
        for(int i=0; i<n_rows; ++i){
          double product = products[i][0]*products[i][1]*products[i][2];
          cout << "after mult many" << endl;
          to_sum.emplace_back(product);
        }
        double result = 0;
        for (int j=0; j < to_sum.size(); ++j){
          result += to_sum[j];
        }
        return result;
        //Matrix adj_T = Matrix(result);
        //return adj_T.get_transpose();
      }

      Matrix (std::vector< std::vector<double> > data);
      std::vector< std::vector<double> > values;
      int n_rows;
      int n_cols;
    
        
   private:
        vector< vector<double>> initialize_empty(int rows, int cols){
            double ZERO = 0;
            vector< vector<double> > result;
            for (int i=0; i < rows; ++i){
                std::vector<double> row_result;
                for (int j=0; j < cols; ++j){
                    row_result.emplace_back(ZERO);
                }
                result.emplace_back(row_result);
            }
            return result;
      }
};

Matrix::Matrix (std::vector< std::vector<double> > data) {
  values = data;
  n_rows = data.size();
  n_cols = data[0].size();

}




int main()
{
    // Example: Basics

    example_lr();

    // Wait for ENTER before closing screen.
    cout << "Press ENTER to exit" << endl;
    char ignore;
    cin.get(ignore);

    return 0;
}

Matrix read_data_file(string & file_name, bool is_vector){
    cout << "About to get file data " << file_name << "..." <<endl;
    string line;
    ifstream myfile (file_name);
    vector< vector<double>> matrix_data; 
    if (myfile.is_open())
    {
        while ( getline (myfile,line) ){
            //cout <<"reading" <<endl;
            int start = 0; 
            int first_tab = line.find("\t");
            int next_tab = first_tab;
            int end = line.find("\n");
            if (end == -1){
                end = line.length();
            }
            vector<double> row_data;
            //int last_tab = line.find("\t", first_tab+2);

            if (is_vector){
                double i_0= atof(line.c_str()); 
                row_data.emplace_back(i_0);
                matrix_data.emplace_back(row_data);
            }
            else{
                while (start != line.length()){
                    if (next_tab == -1){
                        next_tab = line.length();
                    }
                    //cout << start << endl;
                    //cout << next_tab << endl;
                    string i_str = line.substr(start, next_tab);
                    double i = atof(i_str.c_str());
                    //cout << "size: " << final_i.size() << endl;
                    row_data.emplace_back(i);
                    start = next_tab;
                    next_tab = line.find("\t", start+1);
                }
                matrix_data.emplace_back(row_data);
            }
            //c = atoi(b.c_str());
            //cout << line << '\n';
        }
        myfile.close();
    }
    else cout << "Unable to open file" << file_name << endl;
    Matrix mat = Matrix(matrix_data);
    //mat.print_decrypted();
    return mat; 
}

void do_lr(Matrix X, Matrix y){
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    cout << "Computing (X^T * X)^(-1) * X^(T)y..." << endl;
    //Model= (X^T * X)^(-1) * X^(T)y

    //Part 1 X^(T)y
    cout << "Computing X^(T)y" <<endl;
    /*
    double det = X.get_determinant();
    cout << "determinant: " << det <<endl;
    Matrix m = X.get_adjugate();
    m.print_decrypted();
    */
    
    Matrix X_T = X.get_transpose();
    Matrix X_T_y = X_T.multiply(y);
    X_T_y.print_decrypted();
    cout << "Done..." << endl <<endl <<endl;


    cout << "Computing (X^T * X)^(-1)" <<endl;
    cout << "(X^T * X)^(-1) = (1/(det(X^T*X)) * Adj(X^T *X)" << endl;
    //Fomula X^T * X)^(-1) = (1/(det(X^T*X)) * Adj(X^T *X)

    cout << "Computing X^T*X" <<endl;
    Matrix X_T_X = X_T.multiply(X);
    X_T_X.print_decrypted();

    cout << "Computing Adj(X^T *X)" <<endl;
    Matrix adj_xtx = X_T_X.get_adjugate();
    adj_xtx.print_decrypted();

    cout << "Computing det(X^T*x)"<<endl;
    double det = X_T_X.get_determinant();

    cout << "Done..." <<endl <<endl <<endl;

    cout << "Last step computing Adj(X^T *X) * X^(T)y" << endl;

    Matrix final = adj_xtx.multiply(X_T_y);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "LR took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds\n";

    cout << "Done with linear Regression!" << endl <<endl <<endl;

    cout << "Matrix: " << endl;
    final.print_decrypted();
    cout << endl <<endl;

    cout << "1/determinant multiple: " << endl;
    cout << det << endl;
    
}

void example_lr()
{
    print_example_banner("Linear Regression DEGREE 3");
    /*
    string file_name_data_5  = "../sealcrypto/data_5_3.txt";
    string file_name_labels_5 = "../sealcrypto/labels_5_3.txt";
    Matrix X_5 = read_data_file(file_name_data_5, false);
    Matrix y_5 = read_data_file(file_name_labels_5, true);

    cout << "5 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_5, y_5);

    string file_name_data_10  = "../sealcrypto/data_10_3.txt";
    string file_name_labels_10 = "../sealcrypto/labels_10_3.txt";
    Matrix X_10 = read_data_file(file_name_data_10, false);
    Matrix y_10 = read_data_file(file_name_labels_10, true);
    
    cout << "10 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_10, y_10);

    string file_name_data_25  = "../sealcrypto/data_25_3.txt";
    string file_name_labels_25 = "../sealcrypto/labels_25_3.txt";
    Matrix X_25 = read_data_file(file_name_data_25, false);
    Matrix y_25 = read_data_file(file_name_labels_25, true);
    
    cout << "25 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_25, y_25);

    string file_name_data_50  = "../sealcrypto/data_50_3.txt";
    string file_name_labels_50 = "../sealcrypto/labels_50_3.txt";
    Matrix X_50 = read_data_file(file_name_data_50, false);
    Matrix y_50 = read_data_file(file_name_labels_50, true);
    
    cout << "50 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_50, y_50);

    string file_name_data_100  = "../sealcrypto/data_100_3.txt";
    string file_name_labels_100 = "../sealcrypto/labels_100_3.txt";
    Matrix X_100 = read_data_file(file_name_data_100, false);
    Matrix y_100 = read_data_file(file_name_labels_100, true);
    
    cout << "100 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_100, y_100);
    */
    string file_name_data_200  = "../sealcrypto/data_200_3.txt";
    string file_name_labels_200 = "../sealcrypto/labels_200_3.txt";
    Matrix X_200 = read_data_file(file_name_data_200, false);
    Matrix y_200 = read_data_file(file_name_labels_200, true);
    
    
    cout << "200 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_200, y_200);


    string file_name_data_500  = "../sealcrypto/data_500_3.txt";
    string file_name_labels_500 = "../sealcrypto/labels_500_3.txt";
    Matrix X_500 = read_data_file(file_name_data_500, false);
    Matrix y_500 = read_data_file(file_name_labels_500, true);
    
    cout << "500 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_500, y_500);

    string file_name_data_1000  = "../sealcrypto/data_1000_3.txt";
    string file_name_labels_1000 = "../sealcrypto/labels_1000_3.txt";
    Matrix X_1000 = read_data_file(file_name_data_1000, false);
    Matrix y_1000 = read_data_file(file_name_labels_1000, true);
    
    cout << "1000 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_1000, y_1000);
    
}

void print_example_banner(string title)
{
    if (!title.empty())
    {
        size_t title_length = title.length();
        size_t banner_length = title_length + 2 + 2 * 10;
        string banner_top(banner_length, '*');
        string banner_middle = string(10, '*') + " " + title + " " + string(10, '*');

        cout << endl
            << banner_top << endl
            << banner_middle << endl
            << banner_top << endl
            << endl;
    }
}