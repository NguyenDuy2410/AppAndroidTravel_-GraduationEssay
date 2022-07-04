/*
 * Created by ishaanjav
 * github.com/ishaanjav
 */

package app.ij.mlwithtensorflowlite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Model;


public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 140;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 140, 140, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f /255.f ));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Bảo tàng Chiến dịch Hồ Chí Minh", "Bưu Điện HCM", "Bảo Tàng Tranh 3D Art In Us", "BiTexco", "Bến Cảng Nhà Rồng", "Bảo tàng Chứng tích Chiến tranh TPHCM", "Bảo Tàng Mỹ Thuật TPHCM", "Bảo tàng Phụ Nữ Nam Bộ", "Bảo tàng Thành phố Hồ Chí Minh", "Bảo Tàng Lịch Sử VN TPHCM", "Chùa Bửu Long", "Chùa Hoằng Pháp", "Chùa Pháp Bửu", "chùa ngọc hoàng", "Chùa Quốc Tự", "Chùa Chantarangsay(Chùa Khmer)", "Bến tàu Bạch Đằng", "Chùa Huê Nghiêm Q2", "Chùa Một Cột Thủ Đức", "Chùa Bà Thiên Hậu Sài Gòn", "Chợ Đêm Hạnh Thông Tây", "Chợ Tân Định", "Chùa Ông Bổn", "Chợ Lớn (Chợ Bình Tây)", "Chợ bà chiểu", "Chùa Ấn Quang", "Chợ Bến Thành", "Chợ Kim Biên", "Công viên giải trí Đầm Sen", "Chùa Vạn Đức", "công viên Vinhomes Central Park", "Công Viên Suối Tiên", "Dinh Độc Lập", "công viên thỏ trắng", "Hồ Con Rùa TPHCM", "Landmark 81", "Công viên đá Nhật Rinrin Park", "Cầu Ánh Sao Sài Gòn", "Hẻm Bia Lost in HongKong", "Pháp Viện Minh Đăng Quang", "nhà thờ hạnh thông tây", "Phù Châu Miếu (Miếu Nổi)", "Nhà Thờ Tân Định", "Nhà Thờ Cha Tam", "Nowzone Fashion Mall Shopping Center", "Nhà thờ Huyện Sĩ", "Nhà thờ Chợ Quán", "Nhà Thờ Đức Bà", "Nhà Hát TPHCM", "Tu Viện VĨNH NGHIÊM", "Đền Mariamman", "Thảo Cầm Viên SG", "Phố Tây Bùi Viện TPHCM", "Đường sách Nguyên Văn Bình", "Đài tưởng niệm Bồ tát Thích Quảng Đức", "Áo Dài Exhibition", "Phố Đi Bộ Nguyễn Huệ", "Tu Viện Khánh An", "Đền Tưởng Niệm Bến Dược", "Địa đạo Củ Chi"};
            result.setText(classes[maxPos]);
            result.setOnClickListener(new View.OnClickListener()
                  {
                      @Override
                      public void onClick(View view){
                          startActivity(new Intent(Intent.ACTION_VIEW,
                                  Uri.parse("https://vi.wikipedia.org/wiki/"+result.getText())));
                      }
                  }
            );
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}