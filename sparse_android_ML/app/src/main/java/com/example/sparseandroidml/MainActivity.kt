package com.example.sparseandroidml

import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.Manifest.permission.WRITE_EXTERNAL_STORAGE
import android.R
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.audiofx.BassBoost
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.service.autofill.OnClickAction
import android.util.Log
import android.view.MotionEvent
import android.view.View.OnClickListener
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
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import org.checkerframework.checker.signedness.qual.Unsigned
import java.text.DecimalFormat
import kotlin.system.measureNanoTime

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var _runs: Int = 1

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
            
            if (true) //(!outFile.exists())
            {
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

    fun rescaleAndConvertToMonochrome(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        // rescale to width x height
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)

        // convert to monochrome
        val monochromeBitmap = Bitmap.createBitmap(width, height, resizedBitmap.config)

        // color filter for monochrome conversion
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val colorFilter = ColorMatrixColorFilter(colorMatrix)

        val canvas = Canvas(monochromeBitmap)
        val paint = Paint()
        paint.colorFilter = colorFilter

        // draw the monochrome bitmap
        canvas.drawBitmap(resizedBitmap, 0f, 0f, paint)

        return monochromeBitmap
    }

    fun getPixelArray(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixelArray = IntArray(width * height)

        // get the pixel values as an IntArray
        bitmap.getPixels(pixelArray, 0, width, 0, 0, width, height)

        // convert from Int to Float, normalize between 0 and 1
        val floatArray = FloatArray(pixelArray.size)
        for (i in pixelArray.indices) {
            floatArray[i] = Color.red(pixelArray[i]) / 255.0f
        }

        return floatArray
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        loadModels()

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

        /*saveConfigToInternalStorage()

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

        binding.sampleText.text = "Elapsed time TFLite: $elapsed ms\n" + stringFromJNI()*/

        val model = Mnist.newInstance(this)
        val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 1024), DataType.FLOAT32)

        with (binding.clearBtn) {
            setOnClickListener {
                binding.digitDrawView.clearCanvas()
                binding.resultText.text = ""
                binding.performanceAverageText.text = ""
                binding.performanceTotalText.text = ""
            }
        }

        with (binding.classifyTFBtn) {
           setOnClickListener {
               val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))

               var predictedIndex: Int? = 0
               val elapsed = measureNanoTime {
                   for (i in 0 until _runs)
                   {
                       inputFeature.loadArray(pixelArray)
                       val modelOutput = model.process(inputFeature)
                       val outputFeature = modelOutput.outputFeature0AsTensorBuffer
                       predictedIndex = outputFeature.floatArray.withIndex().maxByOrNull { it.value }?.index
                   }
               }
               val formater = DecimalFormat("#.###")
               binding.resultText.text = predictedIndex.toString()
               binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"
               binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
           }
        }

        with (binding.classify4in16Btn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))

                var predictedIndex : Int = 0
                val elapsed = measureNanoTime {
                    for (i in 0 until _runs)
                    {
                        predictedIndex = run4in16model(pixelArray)
                    }
                }
                val formater = DecimalFormat("#.###")
                binding.resultText.text = predictedIndex.toString()
                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"
                binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
            }
        }

        with (binding.classify2in16Btn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))

                var predictedIndex : Int = 0
                val elapsed = measureNanoTime {
                    for (i in 0 until _runs)
                    {
                        predictedIndex = run2in16model(pixelArray)
                    }
                }
                val formater = DecimalFormat("#.###")
                binding.resultText.text = predictedIndex.toString()
                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"
                binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
            }
        }

        with (binding.digitDrawView) {
            setStrokeWidth(70f)
            setColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
        }

        with (binding.subtractRunBtn) {
            setOnClickListener {
                if (_runs > 1)
                {
                    _runs--
                    binding.runsText.text = "Runs: $_runs"
                }
            }
        }

        with(binding.addRunBtn) {
            setOnClickListener {
                _runs++
                binding.runsText.text = "Runs: $_runs"
            }
        }
    }

    external fun loadModels(): Unit
    external fun run4in16model(sample: FloatArray): Int
    external fun run2in16model(sample: FloatArray): Int

    companion object {
        // Used to load the 'sparseandroidml' library on application startup.
        init {
            System.loadLibrary("sparseandroidml")
        }
    }
}