from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk menghitung jarak antara dua titik koordinat
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius bumi dalam kilometer

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

#model candi


#    Baca data UMKM
umkm_data = pd.read_csv('umkm.csv')

def get_umkm_recommendations(user_lat, user_lon):
    # Hitung jarak antara user dan setiap UMKM
    umkm_data['distance'] = umkm_data.apply(
        lambda row: haversine_distance(user_lat, user_lon, row['Latitude_umkm'], row['Longitude_umkm']), 
        axis=1
    )

    # Filter UMKM dalam radius 1 km
    nearby_umkm = umkm_data[umkm_data['distance'] <= 1].copy()

    if nearby_umkm.empty:
        return "Tidak ada UMKM dalam radius 1 km."

    # Buat model regresi linear
    X = nearby_umkm[['distance']]
    y = nearby_umkm['Rating_umkm']

    model = LinearRegression()
    model.fit(X, y)

    # Prediksi rating untuk UMKM terdekat
    nearby_umkm['predicted_rating'] = model.predict(X)

    # Hitung MSE
    mse = mean_squared_error(nearby_umkm['Rating_umkm'], nearby_umkm['predicted_rating'])
    print(f'Mean Squared Error: {mse}')

    # Urutkan berdasarkan rating prediksi
    recommendations = nearby_umkm.sort_values('predicted_rating', ascending=False)

    return recommendations[['Nama UMKM', 'Gambar_UMKM']].head(5)

#model candi
# Baca data dari file CSV 
candi_df = pd.read_csv('candi.csv')
user_df = pd.read_csv('user.csv')

# Filter kolom yang diperlukan
candi_filtered_df = candi_df[['place_id', 'Nama_Candi', 'Deskripsi', 'Rating_candi', 'Gambar_Candi']]
user_filtered_df = user_df[['user_id', 'place_id', 'user_rating']]

# Buat matriks user-item
user_item_matrix = user_filtered_df.pivot_table(index='user_id', columns='place_id', values='user_rating', fill_value=0)

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit dan transform deskripsi candi
tfidf_matrix = tfidf_vectorizer.fit_transform(candi_filtered_df['Deskripsi'])

# Hitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Latih model SVD (pastikan n_components sesuai dengan jumlah candi unik)
n_components = len(user_item_matrix.columns)
svd = TruncatedSVD(n_components=n_components) 
matrix_factorized = svd.fit_transform(user_item_matrix)


# 3. Fungsi untuk mendapatkan rekomendasi hibrida (disesuaikan)
def get_hybrid_recommendations(user_id, matrix_factorized, cosine_sim, candi_filtered_df, num_recommendations=5):
    try:
        # Dapatkan prediksi rating untuk pengguna
        user_index = user_item_matrix.index.get_loc(user_id)
        predicted_ratings = matrix_factorized[user_index]

        # Dapatkan indeks item yang sudah diberi rating
        rated_item_indices = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index

        # Set prediksi rating item yang sudah diberi rating menjadi -1
        predicted_ratings[rated_item_indices] = -1

        # Dapatkan indeks top N berdasarkan prediksi rating
        top_n_indices = predicted_ratings.argsort()[-num_recommendations:][::-1]

        # Buat DataFrame dengan place_id dan predicted_rating
        top_n = pd.DataFrame({'place_id': top_n_indices, 'predicted_rating': predicted_ratings[top_n_indices]})

        # Hitung kemiripan item untuk item yang direkomendasikan teratas
        item_similarities = cosine_sim[top_n_indices]

        # Ubah bentuk predicted_rating agar kompatibel untuk perkalian
        predicted_ratings_reshaped = predicted_ratings[top_n_indices].reshape(-1, 1)

        # Hitung skor hibrida
        top_n['hybrid_score'] = (item_similarities * predicted_ratings_reshaped).sum(axis=1) / item_similarities.sum(axis=1)

        # Urutkan berdasarkan hybrid_score dan kembalikan rekomendasi teratas
        top_n = top_n.sort_values(by='hybrid_score', ascending=False)

        # Gabungkan dengan data asli untuk mendapatkan Nama_Candi dan Gambar_Candi
        top_n = top_n.merge(candi_filtered_df[['place_id', 'Nama_Candi', 'Gambar_Candi']], on='place_id')

        # Kembalikan rekomendasi teratas dengan Nama_Candi, hybrid_score, dan Gambar_Candi
        return top_n[['Nama_Candi', 'hybrid_score', 'Gambar_Candi']].head(num_recommendations)

    except ValueError as e:
        if "not in index" in str(e):
            return jsonify({"message": f"Error: User ID {user_id} not found in training data."}), 404  # Not Found
        else:
            return jsonify({"message": f"An unexpected error occurred: {str(e)}"}), 500  # Internal Server Error

    except Exception as e:
        return jsonify({"message": f"An unexpected error occurred: {str(e)}"}), 500  # Internal Server Error

# Buat aplikasi Flask
app = Flask(__name__)


# 4. Flask endpoint (diperbaiki)
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    user_id_str = request.args.get('user_id')

    try:
        user_id = int(user_id_str) if user_id_str else 1  # Gunakan user_id 1 sebagai default jika tidak diberikan
    except ValueError:
        return jsonify({"message": "Error: 'user_id' must be an integer."}), 400  # Bad Request

    recommendations = get_hybrid_recommendations(user_id, matrix_factorized, cosine_sim, candi_filtered_df)

    if not recommendations.empty:
        # Ubah nama kolom 'Gambar_Candi' menjadi 'thumbnail' 
        recommendations = recommendations.rename(columns={
            'Gambar_Candi': 'thumbnail',
            'Nama_Candi': 'name'
            })

        # Bungkus rekomendasi dalam objek CandiResponse
        response_data = {
            "data": recommendations.to_dict(orient='records'),
            "status": "S",
            "message": "Success"
        }
        return jsonify(response_data)
    else:
        return jsonify({"message": "No recommendations found for this user."}), 404 

@app.route('/recommendation-wisata-kuliner', methods=['GET'])
def get_recommendations_api():
    try:
        user_lat = request.args.get('latitude')
        user_lon = request.args.get('longitude')

        if user_lat and user_lon:
            user_lat = float(user_lat)
            user_lon = float(user_lon)
            recommendations = get_umkm_recommendations(user_lat, user_lon)
        else:
            # Berikan rekomendasi default jika tidak ada input lokasi
            recommendations = umkm_data.sort_values('Rating_umkm', ascending=False).head(5)  # Atau logika rekomendasi default lainnya

        if not recommendations.empty:
            # Ubah nama kolom 'Gambar_UMKM' menjadi 'thumbnail' 
            recommendations = recommendations.rename(columns={
                'Gambar_UMKM': 'thumbnail',
                'Nama UMKM': 'name',
                'Provinsi': 'location'
                })

            # Bungkus rekomendasi dalam objek CandiResponse
            response_data = {
                "data": recommendations.to_dict(orient='records'),
                "status": "S",
                "message": "Success"
            }
            return jsonify(response_data)
        else:
            return jsonify({"message": "Tidak ada rekomendasi yang ditemukan", "status": "F"})

    except ValueError: 
        return jsonify({"message": "Invalid latitude or longitude parameters", "status": "E"}), 400
    except Exception as e: 
        return jsonify({"message": "An error occurred", "status": "E"}), 500

if __name__ == '__main__':
    app.run(debug=True) 