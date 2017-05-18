#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include "seal.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath> 

using namespace std;
using namespace seal;

void print_example_banner(string title);

void linear_regression();

class Matrix {
   public:
      Matrix add_matrix(Matrix & other, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator) {
         cout << "Adding two matricies together..." <<endl;
         std::vector< std::vector<seal::BigPolyArray> > result;
         for (int i = 0; i < n_rows; ++i)
         {
            std::vector<seal::BigPolyArray> row_result;
            for (int j= 0; j < n_cols; ++j){
               seal::BigPolyArray new_value = evaluator.add(values[i][j], other.values[i][j]);
               row_result.emplace_back(new_value);
            }
            result.emplace_back(row_result);
         }
         return Matrix(result);
      }
      Matrix multiply(Matrix & other, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator) {
         cout << "Multiplying 2 matricies together..."  <<endl;
         int rows = n_rows; 
         int cols = other.n_cols;
         std::vector< std::vector< std::vector<seal::BigPolyArray> > > to_sum = initialize_empty(rows, cols, encoder, decryptor, encryptor);
         for (int i = 0; i < n_rows; ++i)
         {
            for (int j= 0; j < other.n_cols; ++j){
                for (int k=0; k < n_cols; ++k){
                    seal::BigPolyArray product = evaluator.multiply(values[i][k], other.values[k][j]);
                    to_sum[i][j].emplace_back(product);
                }
            }
         }
        std::vector< std::vector<seal::BigPolyArray> > result;
        for (int i = 0; i < n_rows; ++i)
         {
            std::vector<seal::BigPolyArray> row_result;
            for (int j= 0; j < other.n_cols; ++j){
              BigPolyArray sum = evaluator.add_many(to_sum[i][j]);
              row_result.emplace_back(sum);
            }
            result.emplace_back(row_result);
         }
         return Matrix(result);
      }

      Matrix multiply_constant(seal::BigPolyArray & constant, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator) {
         cout << "Mutliplying a matrix by a constant..." <<endl;
         std::vector< std::vector<seal::BigPolyArray> > result;
         for (int i = 0; i < n_rows; ++i)
         {
            std::vector<seal::BigPolyArray> row_result;
            for (int j= 0; j < n_cols; ++j){
               seal::BigPolyArray new_value = evaluator.multiply(values[i][j], constant);
               row_result.emplace_back(new_value);
            }
           result.emplace_back(row_result);
         }
         return Matrix(result);
      }

      Matrix get_transpose(){
         cout << "Calculating the transpose of a matrix..." <<endl;
         std::vector< std::vector<seal::BigPolyArray> > result;
         for (int i = 0; i < n_cols; ++i)
         {
            std::vector<seal::BigPolyArray> row_result;
            for (int j= 0; j < n_rows; ++j){
               seal::BigPolyArray new_value = values[j][i];
               row_result.emplace_back(new_value);
            }
           result.emplace_back(row_result);
         }
         return Matrix(result);
      }

      Matrix get_adjugate(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
        cout << "Calculating the adjugate of a matrix..." <<endl;
        if (n_cols == 1 && n_rows == 1){
          std::vector< std::vector<seal::BigPolyArray> > result;
          double ONE = 1;
          BigPoly encoded_number = encoder.encode(ONE);
          std::vector<seal::BigPolyArray> row = {encryptor.encrypt(encoded_number)};
          result.emplace_back(row);
          return Matrix(result);
        }
        if (n_cols == 2 && n_rows == 2){
            std::vector< std::vector<seal::BigPolyArray> > result = initialize_zero(2, 2, encoder, decryptor, encryptor);
            //for 2x2 matrix [[a,b],[c,d]] adjugate is [[d, -b],[-c, a]]
            result[0][0] = values[1][1];
            result[0][1] = evaluator.negate(values[0][1]);
            result[1][0] = evaluator.negate(values[1][0]);
            result[1][1] = values[0][0];
            return Matrix(result);
        }
        else if (n_cols == 3 && n_rows == 3){
            std::vector< std::vector<BigPolyArray>> m_0_0_values = {{values[1][1], values[1][2]}, {values[2][1], values[2][2]}};
            Matrix m_0_0 =Matrix(m_0_0_values);
            std::vector< std::vector<BigPolyArray>> m_0_1_values = {{values[0][1], values[0][2]}, {values[2][1], values[2][2]}};
            Matrix m_0_1 =Matrix(m_0_1_values);
            std::vector< std::vector<BigPolyArray>> m_0_2_values = {{values[0][1], values[0][2]}, {values[1][1], values[1][2]}};
            Matrix m_0_2 =Matrix(m_0_2_values);
            BigPolyArray a_0_0 = m_0_0.get_determinant(encoder, decryptor, encryptor, evaluator);
            BigPolyArray a_0_1 = evaluator.negate(m_0_1.get_determinant(encoder, decryptor, encryptor, evaluator));
            BigPolyArray a_0_2 = m_0_2.get_determinant(encoder, decryptor, encryptor, evaluator);


            std::vector< std::vector<BigPolyArray>> m_1_0_values = {{values[1][0], values[1][2]}, {values[2][0], values[2][2]}};
            Matrix m_1_0 =Matrix(m_1_0_values);
            std::vector< std::vector<BigPolyArray>> m_1_1_values = {{values[0][0], values[0][2]}, {values[2][0], values[2][2]}};
            Matrix m_1_1 =Matrix(m_1_1_values);
            std::vector< std::vector<BigPolyArray>> m_1_2_values = {{values[0][0], values[0][2]}, {values[1][0], values[1][2]}};
            Matrix m_1_2 =Matrix(m_1_2_values);

            BigPolyArray a_1_0 = evaluator.negate(m_1_0.get_determinant(encoder, decryptor, encryptor, evaluator));
            BigPolyArray a_1_1 = m_1_1.get_determinant(encoder, decryptor, encryptor, evaluator);
            BigPolyArray a_1_2 = evaluator.negate(m_1_2.get_determinant(encoder, decryptor, encryptor, evaluator));

            std::vector< std::vector<BigPolyArray>> m_2_0_values = {{values[1][0], values[1][1]}, {values[2][0], values[2][1]}};
            Matrix m_2_0 =Matrix(m_2_0_values);
            std::vector< std::vector<BigPolyArray>> m_2_1_values = {{values[0][0], values[0][1]}, {values[2][0], values[2][1]}};
            Matrix m_2_1 =Matrix(m_2_1_values);
            std::vector< std::vector<BigPolyArray>> m_2_2_values = {{values[0][0], values[0][1]}, {values[1][0], values[1][1]}};
            Matrix m_2_2 =Matrix(m_2_2_values);

            BigPolyArray a_2_0 = m_2_0.get_determinant(encoder, decryptor, encryptor, evaluator);
            BigPolyArray a_2_1 = evaluator.negate(m_2_1.get_determinant(encoder, decryptor, encryptor, evaluator));
            BigPolyArray a_2_2 = m_2_2.get_determinant(encoder, decryptor, encryptor, evaluator);
            
            std::vector< std::vector<BigPolyArray>> result = {{a_0_0, a_0_1, a_0_2}, {a_1_0, a_1_1, a_1_2}, {a_2_0, a_2_1, a_2_2}};

            return Matrix(result);
        }
        else{
            return get_adjugate_recursive(encoder, decryptor, encryptor, evaluator);
        }

      }

      Matrix get_adjugate_recursive(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
        cout << "Calculating the adjugate of a matrix..." <<endl;
        std::vector< std::vector<seal::BigPolyArray> > result;
        for(int i=0; i < n_rows; ++i){
          std::vector<seal::BigPolyArray> row_result;
          for(int j=0; j < n_cols; ++j){
            Matrix m = wo_row_col(i, j);
            int sign = std::pow(-1,i+j);
            BigPolyArray det = m.get_determinant(encoder, decryptor, encryptor, evaluator);
            BigPolyArray value = evaluator.multiply_plain(det, encoder.encode(sign));
            row_result.emplace_back(value);
          }
          result.emplace_back(row_result);
        }
        Matrix adj_T = Matrix(result);
        return adj_T.get_transpose();
      }

      Matrix wo_row_col(int row, int col){
        std::vector< std::vector<seal::BigPolyArray> > result;
        for(int i=0; i < n_rows; ++i){
          if(i != row){
            std::vector<seal::BigPolyArray> row_result;
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
 
      BigPolyArray get_determinant(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
        if (n_cols == 1 && n_rows == 1){
          return values[0][0];
        }
        if (n_cols == 2 && n_rows == 2){
            cout << "calculating determinat of a 2x2 matrix" <<endl;
            BigPolyArray ac = evaluator.multiply(values[0][0], values[1][1]);
            BigPolyArray bd = evaluator.multiply(values[0][1], values[1][0]);

            double ac_decode = encoder.decode(decryptor.decrypt(ac));
            double bd_decode = encoder.decode(decryptor.decrypt(bd));
            BigPolyArray det = evaluator.sub(ac, bd);
            return det;
        }
        else if (n_cols == 3 && n_rows == 3){
            cout << "calculating determinat of a 3x3 matrix" <<endl;
            BigPolyArray a = values[0][0];
            BigPolyArray b = values[0][1];
            BigPolyArray c = values[0][2];
            BigPolyArray d = values[1][0];
            BigPolyArray e = values[1][1];
            BigPolyArray f = values[1][2];
            BigPolyArray g = values[2][0];
            BigPolyArray h = values[2][1];
            BigPolyArray i = values[2][2];
            std::vector<BigPolyArray> aei_vec = {a, e, i};
            std::vector<BigPolyArray> bfg_vec = {b, f, g};
            std::vector<BigPolyArray> cdh_vec = {c, d, h};
            std::vector<BigPolyArray> ceg_vec = {c, e, g};
            std::vector<BigPolyArray> bdi_vec = {b, d, i};
            std::vector<BigPolyArray> afh_vec = {a, f, h};
            BigPolyArray aei = evaluator.multiply_many(aei_vec);
            BigPolyArray bfg = evaluator.multiply_many(bfg_vec);
            BigPolyArray cdh = evaluator.multiply_many(cdh_vec);
            BigPolyArray ceg = evaluator.negate(evaluator.multiply_many(ceg_vec));
            BigPolyArray bdi = evaluator.negate(evaluator.multiply_many(bdi_vec));
            BigPolyArray afh = evaluator.negate(evaluator.multiply_many(afh_vec));

            std::vector<BigPolyArray> det_to_sum = {aei, bfg, cdh, ceg, bdi, afh};
            BigPolyArray det = evaluator.add_many(det_to_sum);
            return det;
        }

        else{
            return get_determinant_recursive(encoder, decryptor, encryptor, evaluator);
        }
      }
      BigPolyArray get_determinant_recursive(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
        if(n_cols != n_rows){
          cout << "NOT A SQUARE MATRIX NO DETERMINANT";
          cout << n_cols << "\t" << n_rows << endl;
        }
        std::vector< std::vector<seal::BigPolyArray> > products;
        int i=1;
        for(int j=0; j < n_cols; ++j){
          Matrix m = wo_row_col(i, j);
          BigPoly sign = encoder.encode(std::pow(-1,i+j));
          BigPolyArray sign_encrypt = encryptor.encrypt(sign);
          BigPolyArray det = m.get_determinant(encoder, decryptor, encryptor, evaluator);
          std::vector<seal::BigPolyArray> to_mul = {values[i][j], det, sign_encrypt};
          products.emplace_back(to_mul);
        }
        std::vector<seal::BigPolyArray> to_sum;
        for(int i=0; i<n_rows; ++i){
          BigPolyArray product = evaluator.multiply_many(products[i]);
          to_sum.emplace_back(product);
        }
        BigPolyArray result = evaluator.add_many(to_sum);
        return result;
      }
      double get_average_noise(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
        double sum = 0;
        for(int row =0; row< n_rows; ++row){
          for(int col=0; col<n_cols; ++col){
            sum += decryptor.inherent_noise_bits(values[row][col]);
          }
        }
        double avg_noise = sum/(n_rows*n_cols);
        return avg_noise;
      }

      double get_max_noise(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
        double max = 0;
        for(int row =0; row< n_rows; ++row){
          for(int col=0; col<n_cols; ++col){
            double noise = decryptor.inherent_noise_bits(values[row][col]);
            if (noise > max){
              max = noise;
            }
          }
        }
        return max;
      }
      void print_decrypted(seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator){
         cout << "Printing matrix..." <<endl;
         for (int i = 0; i < n_rows; ++i)
         {
            for (int j= 0; j < n_cols; ++j){
               seal::BigPoly plain_result = decryptor.decrypt(values[i][j]);
               double result = encoder.decode(plain_result);
               cout << result;
               cout << "\t";
            }
            cout << endl;
         }
      }
      Matrix (std::vector< std::vector<seal::BigPolyArray> > data);
      std::vector< std::vector<seal::BigPolyArray> > values;
      int n_rows;
      int n_cols;
    
        
   private:
        std::vector< std::vector<seal::BigPolyArray> > initialize_zero(int rows, int cols, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor){
            double ZERO = 0;
            std::vector< std::vector<seal::BigPolyArray> > result;
            for (int i=0; i < rows; ++i){
                std::vector<seal::BigPolyArray> row_result;
                for (int j=0; j < cols; ++j){
                    BigPoly encoded_number = encoder.encode(ZERO);
                    row_result.emplace_back(encryptor.encrypt(encoded_number));
                }
                result.emplace_back(row_result);
            }
            return result;
      }
      std::vector< std::vector< std::vector<seal::BigPolyArray> > > initialize_empty(int rows, int cols, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor){
            double ZERO = 0;
            std::vector< std::vector< std::vector<seal::BigPolyArray> > > result;
            for (int i=0; i < rows; ++i){
                std::vector< vector<seal::BigPolyArray> > row_result;
                for (int j=0; j < cols; ++j){
                    std::vector<seal::BigPolyArray> col_result = {};
                    row_result.emplace_back(col_result);
                }
                result.emplace_back(row_result);
            }
            return result;
      }
};

Matrix::Matrix (std::vector< std::vector<seal::BigPolyArray> > data) {
  values = data;
  n_rows = data.size();
  n_cols = data[0].size();

}




int main()
{

    linear_regression();

    // Wait for ENTER before closing screen.
    cout << "Press ENTER to exit" << endl;
    char ignore;
    cin.get(ignore);

    return 0;
}

Matrix read_data_file(string & file_name, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator, int dimension){
    cout << "About to get file data " << file_name << "..." <<endl;
    string line;
    ifstream myfile (file_name);
    vector< vector<BigPolyArray>> matrix_data; 
    if (myfile.is_open())
    {
        while ( getline (myfile,line) ){
            int start = 0;
            int first_tab = line.find("\t");
            int next_tab = first_tab;
            int end = line.find("\n");
            if (end == -1){
                end = line.length();
            }
            vector<BigPolyArray> row_data;
            if (dimension == 1){
                double i_0= atof(line.c_str()); 
                BigPoly encoded_i_0 = encoder.encode(i_0);
                row_data.emplace_back(encryptor.encrypt(encoded_i_0));
                matrix_data.emplace_back(row_data);
            }
            else{
                while (start != line.length()){
                    if (next_tab == -1){
                        next_tab = line.length();
                    }
                    string i_str = line.substr(start, next_tab);
                    double i = atof(i_str.c_str());
                    BigPoly encoded_i = encoder.encode(i);
                    BigPolyArray final_i = encryptor.encrypt(encoded_i);
                    row_data.emplace_back(final_i);
                    start = next_tab;
                    next_tab = line.find("\t", start+1);
                }
                matrix_data.emplace_back(row_data);
            }
        }
        myfile.close();
    }
    else cout << "Unable to open file" << file_name << endl;
    Matrix mat = Matrix(matrix_data);
    return mat; 
}

void do_lr(Matrix & X, Matrix & y, seal::FractionalEncoder & encoder, seal::Decryptor & decryptor, seal::Encryptor & encryptor, seal::Evaluator & evaluator, EncryptionParameters & parms){
  auto t1 = std::chrono::high_resolution_clock::now();

  cout << "Computing (X^T * X)^(-1) * X^(T)y..." << endl;

  //Model= (X^T * X)^(-1) * X^(T)y

  //Part 1 X^(T)y

  cout << "Computing X^(T)y" <<endl;
  Matrix X_T = X.get_transpose();
  Matrix X_T_y = X_T.multiply(y, encoder, decryptor, encryptor, evaluator);
  X_T_y.print_decrypted(encoder, decryptor, encryptor, evaluator);
  cout << "Done..." << endl <<endl <<endl;

  cout << "Computing (X^T * X)^(-1)" <<endl;
  cout << "(X^T * X)^(-1) = (1/(det(X^T*X)) * Adj(X^T *X)" << endl;
  //Fomula X^T * X)^(-1) = (1/(det(X^T*X)) * Adj(X^T *X)

  cout << "Computing X^T*X" <<endl;
  Matrix X_T_X = X_T.multiply(X, encoder, decryptor, encryptor, evaluator);
  X_T_X.print_decrypted(encoder, decryptor, encryptor, evaluator);

  cout << "Computing Adj(X^T *X)" <<endl;
  Matrix adj_xtx = X_T_X.get_adjugate(encoder, decryptor, encryptor, evaluator);
  adj_xtx.print_decrypted(encoder, decryptor, encryptor, evaluator);

  cout << "Computing det(X^T*x)"<<endl;
  BigPolyArray det = X_T_X.get_determinant(encoder, decryptor, encryptor, evaluator);

  cout << "Done..." <<endl <<endl <<endl;

  cout << "Last step computing Adj(X^T *X) * X^(T)y" << endl;

  Matrix final = adj_xtx.multiply(X_T_y, encoder, decryptor, encryptor, evaluator);

  cout << "Done with linear Regression!" << endl <<endl <<endl;

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "LR took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds\n";

  cout << "Matrix: " << endl;
  final.print_decrypted(encoder, decryptor, encryptor, evaluator);
  cout << endl <<endl;

  cout << "1/determinant multiple: " << endl;
  BigPoly plain_det = decryptor.decrypt(det);
  double decoded_det = encoder.decode(plain_det);
  cout << decoded_det << endl;


  int max_noise_bit_count = parms.inherent_noise_bits_max();
  cout << "Noise in encryption of determinant " << decoded_det << ": " << decryptor.inherent_noise_bits(det)
      << "/" << max_noise_bit_count << " bits" << endl;

  double avg_noise_bits = final.get_average_noise(encoder, decryptor, encryptor, evaluator);
  cout << "Avg Noise in encryption of matrix : " << avg_noise_bits
      << "/" << max_noise_bit_count << " bits" << endl;
  double max_matrix_noise_bits = final.get_max_noise(encoder, decryptor, encryptor, evaluator);
  cout << "Max Noise in encryption of matrix : " << max_matrix_noise_bits
      << "/" << max_noise_bit_count << " bits" << endl;

}


void linear_regression()
{
    print_example_banner("Linear Regression DEGREE 3");

    // Create encryption parameters
    EncryptionParameters parms;


    //"1x^8192 + 1" for degree 1,2 (<= 100 data points)
    //"1x^16384 + 1" for degree 4, 3 (>100 data points)
    parms.poly_modulus() = "1x^16384 + 1";
    //ChooserEvaluator::default_parameter_options().at(16384) for degree 4, 3 (>100 data points)
    //ChooserEvaluator::default_parameter_options().at(8192) for degree 1,2 (<= 100 data points) 
    parms.coeff_modulus() = ChooserEvaluator::default_parameter_options().at(16384);

    //30 for degree 4, 3 (>100 data points)
    //23 for degree 1,2 (<= 100 data points)
    parms.plain_modulus() = 1 << 30;

    //24 for degree 1,2 (<= 100 data points)
    //44 for degree 4, 3 (>100 data points)
    parms.decomposition_bit_count() = 44;

    // Generate keys.
    cout << "Generating keys ..." << endl;
    KeyGenerator generator(parms);
    generator.generate(20);
    cout << "... key generation complete" << endl <<endl;
    BigPolyArray public_key = generator.public_key();
    BigPoly secret_key = generator.secret_key();
    EvaluationKeys evaluation_keys = generator.evaluation_keys();

    /*
    We will need a fractional encoder for dealing with the rational numbers. Here we reserve 
    64 coefficients of the polynomial for the integral part (low-degree terms) and expand the 
    fractional part to 32 terms of precision (base 3) (high-degree terms).
    */
    FractionalEncoder encoder(parms.plain_modulus(), parms.poly_modulus(), 64, 32, 3);

    // Create the rest of the tools
    Encryptor encryptor(parms, public_key);
    Evaluator evaluator(parms, evaluation_keys);
    Decryptor decryptor(parms, secret_key);


    string file_name_data_5  = "../data_5_3.txt";
    string file_name_labels_5 = "../labels_5_3.txt";
    Matrix X_5 = read_data_file(file_name_data_5, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_5 = read_data_file(file_name_labels_5, encoder, decryptor, encryptor, evaluator, true);

    cout << "5 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_5, y_5, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_10  = "../data_10_3.txt";
    string file_name_labels_10 = "../labels_10_3.txt";
    Matrix X_10 = read_data_file(file_name_data_10, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_10 = read_data_file(file_name_labels_10, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "10 data points " << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_10, y_10, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_25  = "../data_25_3.txt";
    string file_name_labels_25 = "../labels_25_3.txt";
    Matrix X_25 = read_data_file(file_name_data_25, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_25 = read_data_file(file_name_labels_25, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "25 data points " << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_25, y_25, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_50  = "../data_50_3.txt";
    string file_name_labels_50 = "../labels_50_3.txt";
    Matrix X_50 = read_data_file(file_name_data_50, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_50 = read_data_file(file_name_labels_50, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "50 data points with degree 2" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_50, y_50, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_100  = "../data_100_3.txt";
    string file_name_labels_100 = "../labels_100_3.txt";
    Matrix X_100 = read_data_file(file_name_data_100, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_100 = read_data_file(file_name_labels_100, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "100 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_100, y_100, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_200  = "../data_200_3.txt";
    string file_name_labels_200 = "../labels_200_3.txt";
    Matrix X_200 = read_data_file(file_name_data_200, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_200 = read_data_file(file_name_labels_200, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "200 data points " << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_200, y_200, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_500  = "../data_500_3.txt";
    string file_name_labels_500 = "../labels_500_3.txt";
    Matrix X_500 = read_data_file(file_name_data_500, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_500 = read_data_file(file_name_labels_500, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "500 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_500, y_500, encoder, decryptor, encryptor, evaluator, parms);

    string file_name_data_1000  = "../data_1000_3.txt";
    string file_name_labels_1000 = "../labels_1000_3.txt";
    Matrix X_1000 = read_data_file(file_name_data_1000, encoder, decryptor, encryptor, evaluator, false);
    Matrix y_1000 = read_data_file(file_name_labels_1000, encoder, decryptor, encryptor, evaluator, true);
    
    cout << "1000 data points" << endl;
    cout << "Encrypted data matrix and label vector have been created." << endl;
    cout << "Preparing to do linear regression..." << endl <<endl <<endl;
    do_lr(X_1000, y_1000, encoder, decryptor, encryptor, evaluator, parms);
    
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