package com.face.recognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textview.MaterialTextView;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;

import java.io.ByteArrayOutputStream;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // CAMERA REQUEST CODE
    private static final int CAMERA_REQUEST = 1888;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;

    // initialize views here
    private ImageView mImageViewCapturedImage, mImageViewRegistered;
    private MaterialButton mButtonRegister, mButtonVerify;
    private MaterialTextView mTextViewFaces;

    // Shared Preferences
    private SharedPreferences preferences;
    private SharedPreferences.Editor editor;
    public static final String PREF_KEY= "isARealFace";
    public static final String FACE_KEY= "yesToARealFace";

    // String picture[0]= registeredImage & String picture[1]= capturedImage;
    String [] picture;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();

        picture= new String[2];
        preferences= getSharedPreferences(PREF_KEY, Context.MODE_PRIVATE);
        editor= preferences.edit();

        if(!Python.isStarted()){
            // this will start python
            Python.start(new AndroidPlatform(this));
        }
        final Python python= Python.getInstance();

        mButtonRegister.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_PERMISSION_CODE);
                }
                else{
                    Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, CAMERA_REQUEST);
                }
            }
        });

        mButtonVerify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getImageFromPref();

//                PyObject pyObject= python.getModule("compareImages");
//                PyObject object= pyObject.callAttr("compareImages", picture[0], picture[1]);
            }
        });

    }

    private void initViews(){
        mImageViewCapturedImage= findViewById(R.id.mImageViewCapturedImage);
        mImageViewRegistered= findViewById(R.id.mImageViewRegistered);
        mButtonRegister= findViewById(R.id.mButtonRegister);
        mButtonVerify= findViewById(R.id.mButtonVerify);
        mTextViewFaces= findViewById(R.id.mTextViewFaces);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE)
        {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);


            }
            else
            {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            //photo.compress(Bitmap.CompressFormat.JPEG,100,FileOutputStream(file));
            mImageViewCapturedImage.setImageBitmap(photo);
            detectFace(photo);
        }
    }

    private void detectFace(Bitmap photo){
        FirebaseVisionFaceDetectorOptions options =
                new FirebaseVisionFaceDetectorOptions.Builder()
                        .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
                        .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                        .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                        .setMinFaceSize(0.1f)
                        .build();

        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(photo);
        FirebaseVisionFaceDetector detector = FirebaseVision.getInstance().getVisionFaceDetector(options);
        detector.detectInImage(image).addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionFace>>() {
            @Override
            public void onSuccess(List<FirebaseVisionFace> faces) {
                if(isAFace(faces)) {
                    mTextViewFaces.setText(runFaceRecog(faces));
                    saveImageBitmap(photo);
                    picture[1]= imageToString(photo);
                }
                else
                    Toast.makeText(MainActivity.this, "Either no face or multiple faces", Toast.LENGTH_SHORT).show();
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure
                    (@NonNull Exception exception) {
                Toast.makeText(MainActivity.this,
                        "Exception", Toast.LENGTH_LONG).show();
            }
        });
    }

    private String runFaceRecog(List<FirebaseVisionFace> faces) {
        StringBuilder result = new StringBuilder();
        float smilingProbability = 0;
        float rightEyeOpenProbability = 0;
        float leftEyeOpenProbability = 0;

        for (FirebaseVisionFace face : faces) {

            if (face.getSmilingProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
                smilingProbability = face.getSmilingProbability();
            }


            if (face.getRightEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
                rightEyeOpenProbability = face.getRightEyeOpenProbability ();
            }


            if (face.getLeftEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
                leftEyeOpenProbability = face.getLeftEyeOpenProbability();
            }

            result.append("Smile: ");

            if (smilingProbability > 0.5) {

                result.append("Yes \nProbability: " + smilingProbability);

            } else {

                result.append("No");
            }

            result.append("\n\nRight eye: ");

            if (rightEyeOpenProbability > 0.5) {
                result.append("Open \nProbability: " + rightEyeOpenProbability);
            } else {
                result.append("Close");
            }

            result.append("\n\nLeft eye: ");

            if (leftEyeOpenProbability > 0.5) {
                result.append("Open \nProbability: " + leftEyeOpenProbability);
            } else {
                result.append("Close");
            }
            result.append("\n\n");
        }
        return result.toString();
    }

    private boolean isAFace(List<FirebaseVisionFace> faces){
        if(faces.size() == 1){
            return true;
        }
        else{
            return false;
        }
    }

    private String imageToString(Bitmap photo){
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        photo.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] b = baos.toByteArray();

        String encodedImage = Base64.encodeToString(b, Base64.DEFAULT);
        return encodedImage;
    }

    private void saveImageBitmap(Bitmap photo) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        photo.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] b = baos.toByteArray();

        String encodedImage = Base64.encodeToString(b, Base64.DEFAULT);

        String previouslyEncodedImage = preferences.getString(FACE_KEY, "");

        Log.e("encodedImage", encodedImage);
        if(previouslyEncodedImage.equalsIgnoreCase("")){
            editor.putString(FACE_KEY, encodedImage);
            editor.commit();
        }
    }

    private void getImageFromPref() {
        String previouslyEncodedImage = preferences.getString(FACE_KEY, "");

        if( !previouslyEncodedImage.equalsIgnoreCase("") ){
            picture[0]= previouslyEncodedImage;
            byte[] b = Base64.decode(previouslyEncodedImage, Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(b, 0, b.length);
            mImageViewRegistered.setImageBitmap(bitmap);
        }
    }

}