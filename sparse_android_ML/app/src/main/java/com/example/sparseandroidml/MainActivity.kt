package com.example.sparseandroidml

import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.Manifest.permission.WRITE_EXTERNAL_STORAGE
import android.R
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.media.audiofx.BassBoost
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.sparseandroidml.databinding.ActivityMainBinding
import com.example.sparseandroidml.ml.Mnist
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import kotlin.system.measureTimeMillis

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private fun loadCsvFile(filePath: String): Array<FloatArray> {
        val rows = mutableListOf<FloatArray>()

        openFileInput(filePath).use { inputStream ->
            InputStreamReader(inputStream).use { streamReader ->
                BufferedReader(streamReader).use { reader ->
                    var line: String? = reader.readLine()
                    while (line != null) {
                        val values = line.split(",").map { it.toFloat() }.toFloatArray()
                        rows.add(values)
                        line = reader.readLine()
                    }
                }
            }
        }

        return rows.toTypedArray()
    }

    private fun saveConfigToInternalStorage() {
        val assetManager = assets
        val filenames = arrayOf("mnist_X_test.csv", "mnist_X_test_T.csv", "biases_l0.csv", "biases_l1.csv", "biases_l2.csv",
                                "biases_l3.csv", "biases_l4.csv", "weights_l0.csv", "weights_l1.csv", "weights_l2.csv",
                                "weights_l3.csv", "weights_l4.csv", "mnist_y_test.csv")
        for (filename in filenames)
        {
            val outFile = File(filesDir, filename)

            if (!outFile.exists()) {
                try {
                    assetManager.open(filename).use { inputStream ->
                        FileOutputStream(outFile).use { outputStream ->
                            val buffer = ByteArray(1024)
                            var read: Int

                            while (inputStream.read(buffer).also { read = it } != -1) {
                                outputStream.write(buffer, 0, read)
                            }

                            Log.d("MainActivity", "Config file copied to internal storage.")
                        }
                    }
                } catch (e: IOException) {
                    Log.e("MainActivity", "Error copying config file to internal storage.", e)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)


        /*var fileInputStream: FileInputStream? = null
        fileInputStream = openFileInput("mnist_y_test.csv")
        var inputStreamReader: InputStreamReader = InputStreamReader(fileInputStream)
        val bufferedReader: BufferedReader = BufferedReader(inputStreamReader)
        val stringBuilder: StringBuilder = StringBuilder()
        var text: String? = null
        while ({ text = bufferedReader.readLine(); text }() != null) {
            stringBuilder.append(text)
        }*/

        saveConfigToInternalStorage()

        val accuracy : Double
        val elapsed = measureTimeMillis {
            val X_test = loadCsvFile("mnist_X_test.csv")
            val y_test = loadCsvFile("mnist_y_test.csv")

            val model = Mnist.newInstance(this)
            var correctPredictions = 0
            val totalSamples = X_test.size

            for (i in X_test.indices) {
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1024), DataType.FLOAT32)
                inputFeature0.loadArray(X_test[i])

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                val predictedIndex = outputFeature0.floatArray.withIndex().maxByOrNull { it.value }?.index
                val trueIndex = y_test[i][0].toInt()

                if (predictedIndex == trueIndex) {
                    correctPredictions++
                }
            }
            model.close()
            accuracy = correctPredictions.toDouble() / totalSamples * 100
        }

        binding.sampleText.text = "Elapsed time TFLite: $elapsed ms\n" + stringFromJNI()
    }

    /**
     * A native method that is implemented by the 'sparseandroidml' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'sparseandroidml' library on application startup.
        init {
            System.loadLibrary("sparseandroidml")
        }
    }
}