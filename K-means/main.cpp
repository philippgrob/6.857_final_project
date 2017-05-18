#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include "seal.h"

using namespace std;
using namespace seal;

void print_example_banner(string title);

void kmean();

int main()
{
    // K-means algorithm
    kmean();

    // Wait for ENTER before closing screen.
    cout << "Press ENTER to exit" << endl;
    char ignore;
    cin.get(ignore);

    return 0;
}









void kmean() {
    print_example_banner("Running K-Means Algorithm");

    // Construct Coordinates
    const vector<pair<double, double>> coordinates = {make_pair(-3, -1), make_pair(-3, 0.5), make_pair(3., 0.7), make_pair(3., 1.7)}; 

    const vector<pair<double, double>> centers = {make_pair(0., 0.), make_pair(6,6)};
    const int NLOOPS = 3;

    // Print coordinates
    //
    for (int i=0; i < coordinates.size(); i++){
        cout << "We have a coordinates with x coordinate equal to " << coordinates.at(i).first << "and y coordinate equal to " << coordinates.at(i).second << endl;
    }

    // Create encryption parameters
    EncryptionParameters parms;

    /*
    We first choose the polynomial modulus. This must be a power-of-2 cyclotomic polynomial,
    i.e. a polynomial of the form "1x^(power-of-2) + 1". We recommend using polynomials of
    degree at least 1024.
    */
    parms.poly_modulus() = "1x^16384 + 1";
    // Use the default values of 16384
    parms.coeff_modulus() = ChooserEvaluator::default_parameter_options().at(16384);
    // Add the plain modulus
    parms.plain_modulus() = 1 << 20;
    // Maximum noise level should polynomial modulus/plain_modulus
    // We encode each number as a polynomial
    // IntegerEncoder encoder(parms.plain_modulus());
    //
    vector<BigPoly> x_coord, y_coord;

    double norm = 1./coordinates.size();


    cout << "Generating keys ..." << endl;
    KeyGenerator generator(parms);
    generator.generate();
    cout << "... key generation complete" << endl;
    BigPolyArray public_key = generator.public_key();
    BigPoly secret_key = generator.secret_key();
    
    // We use a fractional encoder with the first 64 bits for integers and the last 32 bits for the float portion
    FractionalEncoder encoder(parms.plain_modulus(), parms.poly_modulus(), 64, 32, 3);

    cout << "Ecrypted coordinates ...." << endl;
    // Encrypt each coordinate
    for (int i=0; i < coordinates.size(); i++){
        x_coord.push_back(encoder.encode(coordinates.at(i).first));
        y_coord.push_back(encoder.encode(coordinates.at(i).second));
    }

    // Create the actual encryptors
    Encryptor encryptor(parms, public_key);
    Evaluator evaluator(parms);
    Decryptor decryptor(parms, secret_key);

    vector<BigPolyArray> encrypt_x, encrypt_y;

    for (int i=0; i < coordinates.size(); i++) {
        encrypt_x.push_back(encryptor.encrypt(x_coord.at(i)));
        encrypt_y.push_back(encryptor.encrypt(y_coord.at(i)));
    }

    vector<BigPolyArray> encrypt_x_center, encrypt_y_center;

    for (int i=0; i < centers.size(); i++) {
        encrypt_x_center.push_back(encryptor.encrypt(encoder.encode(centers.at(i).first)));
        encrypt_y_center.push_back(encryptor.encrypt(encoder.encode(centers.at(i).second)));
    }

    // We choose not to encrypt the division element
    BigPoly div_by_elem = encoder.encode(norm);

    cout << "Finished adding all cipher text" << endl;
    BigPolyArray x_sum = evaluator.add_many(encrypt_x);
    BigPolyArray y_sum = evaluator.add_many(encrypt_y);

    BigPolyArray x_mean = evaluator.multiply_plain(x_sum, div_by_elem);
    BigPolyArray y_mean = evaluator.multiply_plain(y_sum, div_by_elem);


    /////////// Computation Here
    vector< BigPolyArray> square_x, square_y;

    cout << "Starting to square differences" << endl;
    for (int i=0; i < encrypt_x.size(); i++ ){
        square_x.push_back(evaluator.square(evaluator.sub(encrypt_x.at(i), x_mean)));
        square_y.push_back(evaluator.square(evaluator.sub(encrypt_y.at(i), y_mean)));
    }

    BigPolyArray x_square_sum = evaluator.add_many(square_x);
    BigPolyArray y_square_sum = evaluator.add_many(square_y);

    for (int i = 0; i < NLOOPS; i++ ){
        // This loop will except vectors encrypt_x_center and encrypt_y_center to be a list of computed points and will expect encrypt_x and encrypt_y to be the encrypted x and y coordinates 
        vector<vector<BigPolyArray>> point_distance(encrypt_x.size());
        // Compute the distance of a point from each of the possible centers
        for (int k=0; k < encrypt_x.size(); k++){
           for (int j = 0; j < encrypt_x_center.size(); j++){
               BigPolyArray result_x = evaluator.square(evaluator.sub(encrypt_x[k], encrypt_x_center[j]));
               BigPolyArray result_y = evaluator.square(evaluator.sub(encrypt_y[k],encrypt_y_center[j]));
               point_distance[k].push_back(evaluator.add(result_x, result_y));
           }
           BigPolyArray distance_sum = evaluator.add_many(point_distance[k]);
           double norm = encoder.decode(decryptor.decrypt(distance_sum));
           BigPoly inverse_norm = encoder.encode(1./norm);
           for (int j = 0; j < encrypt_x_center.size(); j++){
               point_distance[k][j] = evaluator.multiply_plain(point_distance[k][j], inverse_norm);
           }
        }

        // Calculate new centers
        for (int k=0; k < encrypt_x_center.size(); k++){
            vector<BigPolyArray> inter_vector_x, inter_vector_y;
            for (int j=0; j < encrypt_x.size(); j++){
                inter_vector_x.push_back(evaluator.multiply(point_distance[j][k], encrypt_x[j]));
                inter_vector_y.push_back(evaluator.multiply(point_distance[j][k], encrypt_y[j]));
            }
            encrypt_x_center[k] = evaluator.add_many(inter_vector_x);
            encrypt_y_center[k] = evaluator.add_many(inter_vector_y);
            cout << "The encrypted center number "<< k << " is with center coordinatex equal to " << encoder.decode(decryptor.decrypt(encrypt_x_center[k])) << " and y coordinate equal to " << encoder.decode(decryptor.decrypt(encrypt_y_center[k])) << " in loop number " << i <<" with noise of "<< decryptor.inherent_noise_bits(encrypt_x_center[k])<< endl;
        }
    }
    



    /////////// Decryption Here

    cout << "New decrypting result";
    BigPoly plain_x = decryptor.decrypt(x_mean);
    BigPoly plain_y = decryptor.decrypt(y_mean);
    BigPoly plain_square_x = decryptor.decrypt(x_square_sum);
    BigPoly plain_square_y = decryptor.decrypt(y_square_sum);

    double mean_x = encoder.decode(plain_x) ;
    double mean_y = encoder.decode(plain_y);
    double x_ssum = encoder.decode(plain_square_x);
    double y_ssum = encoder.decode(plain_square_y);

    cout << "The mean of the x coordinates was "<< mean_x << "and the mean of the y coordinates was "<< mean_y << endl;

    cout << "The square sum of mean difference of the x coordinates was "<< x_ssum << "and square sum of mean of the y coordinates was "<< y_ssum << endl;


    cout << "Noise in the result: " << decryptor.inherent_noise_bits(x_square_sum)
        << "/" << parms.inherent_noise_bits_max() << " bits" << endl;

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
