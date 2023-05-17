package com.example.sparseandroidml

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.sparseandroidml.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.io.BufferedReader
import java.io.File
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import java.text.DecimalFormat
import kotlin.system.measureNanoTime
import android.content.Context
import android.text.Editable
import android.text.TextWatcher
import com.example.sparseandroidml.ml.DefaultMnist
import com.example.sparseandroidml.ml.OptimizedMnist
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlinx.serialization.json.Json
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString

class TFLiteModel(context: Context, modelFileName: String) {
    private var tflite: Interpreter

    private val inputDataType: DataType
    private val outputDataType: DataType

    private val inputShape: IntArray
    private val outputShape: IntArray

    private val inputQuantScale: Float
    private val inputQuantZeroPoint: Int
    private val outputQuantScale: Float
    private val outputQuantZeroPoint: Int

    init {
        tflite = Interpreter(loadModelFile(context, modelFileName))

        val inputTensor = tflite.getInputTensor(0)
        val outputTensor = tflite.getOutputTensor(0)

        inputDataType = inputTensor.dataType()
        outputDataType = outputTensor.dataType()

        inputShape = inputTensor.shape()
        outputShape = outputTensor.shape()

        inputQuantScale = inputTensor.quantizationParams().scale
        inputQuantZeroPoint = inputTensor.quantizationParams().zeroPoint

        outputQuantScale = outputTensor.quantizationParams().scale
        outputQuantZeroPoint = outputTensor.quantizationParams().zeroPoint
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelFileName: String): ByteBuffer {
        val modelPath = context.getFileStreamPath(modelFileName)
        val modelData = Files.readAllBytes(modelPath.toPath())
        val modelBuffer = ByteBuffer.allocateDirect(modelData.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(modelData)
        return modelBuffer
    }

    fun getPixelArray(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixelArray = IntArray(width * height)

        // get the pixel values as an IntArray
        bitmap.getPixels(pixelArray, 0, width, 0, 0, width, height)

        // convert from Int to Float, normalize between 0 and 1
        val byteArray = ByteBuffer.allocateDirect(pixelArray.size * inputDataType.byteSize()).order(ByteOrder.nativeOrder())
        for (i in pixelArray.indices) {
            byteArray.put((((Color.red(pixelArray[i]) / 255.0f) / inputQuantScale) + inputQuantZeroPoint).toInt().toByte())
        }

        return byteArray
    }

    fun getPixelArray(pixelArray: FloatArray): ByteBuffer {

        val byteArray = ByteBuffer.allocateDirect(pixelArray.size * inputDataType.byteSize()).order(ByteOrder.nativeOrder())
        for (i in pixelArray.indices) {
            byteArray.put((((pixelArray[i]) / inputQuantScale) + inputQuantZeroPoint).toInt().toByte())
        }

        return byteArray
    }

    fun runInference(inputByteBuffer: ByteBuffer): Int {
        val outputByteBuffer = ByteBuffer.allocateDirect(outputShape[0] * outputShape[1] * outputDataType.byteSize()).order(ByteOrder.nativeOrder())

        tflite?.run(inputByteBuffer, outputByteBuffer)

        var maxIndex = -1
        var maxValue = Int.MIN_VALUE
        outputByteBuffer.rewind()
        for (i in 0 until outputShape[1]) {
            val quantValue = outputByteBuffer.get().toInt()
            if (quantValue > maxValue) {
                maxValue = quantValue
                maxIndex = i
            }
        }

        return maxIndex
    }

    fun close() {
        tflite?.close()
    }
}

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var quantizedModel: TFLiteModel
    private lateinit var defaultModel: DefaultMnist
    private lateinit var optimizedModel: OptimizedMnist
    private var _runs: Int = 1

    @Serializable
    enum class Modes {
        SAMPLES, TEST_SET, SAMPLED_TEST_SET
    }
    private var _mode: Modes = Modes.SAMPLES

    @Serializable
    data class Measurement(
        val latency: Float,
        val numberOfSamples: Int,
        val predictionType: Modes,
        val modelName: String
    )
    private val _measurements = mutableListOf<Measurement>()


    private fun storeMeasurements() : Unit {
        var oldMeasurements: List<Measurement>
        val fileName = "measurements.json"
        try {
            try {
                val oldMeasurementsStr = this.openFileInput(fileName).bufferedReader().use { it.readText() }
                oldMeasurements = Json.decodeFromString(oldMeasurementsStr)
            }
            catch (e:Exception)
            {
                oldMeasurements = listOf()
            }
            _measurements.addAll(oldMeasurements)
            val outputStreamWriter = OutputStreamWriter(this.openFileOutput(fileName, Context.MODE_PRIVATE))
            outputStreamWriter.use { it.write(Json.encodeToString(_measurements)) }
            _measurements.clear()
        }
        catch (e:Exception)
        {
            val message : String = if (e.message != null) e.message.toString() else "No message"
            Log.d("storeMeasurements", message)
        }
    }

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
                                "weights_l3.csv", "weights_l4.csv", "mnist_y_test.csv", "quantized_mnist.tflite",)
        for (filename in filenames)
        {
            val outFile = File(filesDir, filename)
            
            if (!outFile.exists())
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

        saveConfigToInternalStorage()

        val X_test = loadCsvFile("mnist_X_test.csv")
        val y_test = loadCsvFile("mnist_y_test.csv")
        quantizedModel =  TFLiteModel(this, "quantized_mnist.tflite")
        defaultModel = DefaultMnist.newInstance(this)
        optimizedModel = OptimizedMnist.newInstance(this)
        val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 1024), DataType.FLOAT32)

        with (binding.clearBtn) {
            setOnClickListener {
                binding.digitDrawView.clearCanvas()
                binding.resultText.text = "" //testMLAPI()
                binding.performanceAverageText.text = ""
                binding.performanceTotalText.text = ""
                _measurements.clear()
            }
        }

        with (binding.saveBtn) {
            setOnClickListener {
                storeMeasurements()
            }
        }

        with (binding.classifyTFDefaultBtn) {
           setOnClickListener {
               val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
               val formater = DecimalFormat("#.###")

               var predictedIndex: Int? = 0
               val elapsed = measureNanoTime {
                   if (_mode == Modes.SAMPLES) {
                       for (i in 0 until _runs)
                       {
                           inputFeature.loadArray(pixelArray)
                           val modelOutput = defaultModel.process(inputFeature)
                           val outputFeature = modelOutput.outputFeature0AsTensorBuffer
                           predictedIndex = outputFeature.floatArray.withIndex().maxByOrNull { it.value }?.index
                       }
                       binding.resultText.text = predictedIndex.toString()
                   }
                   else if (_mode == Modes.SAMPLED_TEST_SET) {
                       var correctPredictions : Int = 0
                       binding.resultText.text = "TFDefault..."
                       for (i in X_test.indices)
                       {
                           inputFeature.loadArray(X_test[i])
                           val modelOutput = defaultModel.process(inputFeature)
                           val outputFeature = modelOutput.outputFeature0AsTensorBuffer
                           predictedIndex = outputFeature.floatArray.withIndex().maxByOrNull { it.value }?.index
                           if (predictedIndex == y_test[i][0].toInt()) {
                               correctPredictions++
                           }
                       }
                       binding.resultText.text = "TFDefault: ${formater.format(correctPredictions / 100f)} %"
                   }
               }

               binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

               if (_mode == Modes.SAMPLES) {
                   binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
               }
               else if (_mode == Modes.SAMPLED_TEST_SET) {
                   binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
               }

               _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "TFDefault"))
           }
        }

        with (binding.classifyTFOptimizedBtn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                var predictedIndex: Int? = 0
                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        for (i in 0 until _runs)
                        {
                            inputFeature.loadArray(pixelArray)
                            val modelOutput = optimizedModel.process(inputFeature)
                            val outputFeature = modelOutput.outputFeature0AsTensorBuffer
                            predictedIndex = outputFeature.floatArray.withIndex().maxByOrNull { it.value }?.index
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        var correctPredictions : Int = 0
                        binding.resultText.text = "TFOptimized..."
                        for (i in X_test.indices)
                        {
                            inputFeature.loadArray(X_test[i])
                            val modelOutput = optimizedModel.process(inputFeature)
                            val outputFeature = modelOutput.outputFeature0AsTensorBuffer
                            predictedIndex = outputFeature.floatArray.withIndex().maxByOrNull { it.value }?.index
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "TFOptimized: ${formater.format(correctPredictions / 100f)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else if (_mode == Modes.SAMPLED_TEST_SET) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "TFOptimized"))
            }
        }

        with (binding.classifyTFQuantizedBtn) {
            setOnClickListener {
                val pixelArray = quantizedModel.getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                var predictedIndex: Int? = 0
                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        for (i in 0 until _runs)
                        {
                            predictedIndex = quantizedModel.runInference(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "TFQuantized..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            predictedIndex = quantizedModel.runInference(quantizedModel.getPixelArray(X_test[i]))
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "TFQuantized: ${formater.format(correctPredictions / 100f)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else if (_mode == Modes.SAMPLED_TEST_SET) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "TFQuantized"))
            }
        }

        with (binding.classifyDense) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        var predictedIndex : Int = 0
                        for (i in 0 until _runs)
                        {
                            predictedIndex = runDenseModel(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "Dense..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = runDenseModel(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "Dense: ${formater.format(correctPredictions / 100f)} %"
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "Dense..."
                        val accuracy = runDenseModelTestSet() * 100
                        binding.resultText.text = "Dense: ${formater.format(accuracy)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "Dense"))
            }
        }

        with (binding.classifyDenseThreads) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        var predictedIndex : Int = 0
                        for (i in 0 until _runs)
                        {
                            predictedIndex = runDenseThreadsSample(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "DenseThreads..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = runDenseThreadsSample(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "DenseThreads: ${formater.format(correctPredictions / 100f)} %"
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "DenseThreads..."
                        val accuracy = runDenseThreadsTestSet() * 100
                        binding.resultText.text = "DenseThreads: ${formater.format(accuracy)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "DenseThreads"))
            }
        }

        with (binding.classifyDenseNNAPI) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                var predictedIndex : Int = 0
                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        for (i in 0 until _runs)
                        {
                            predictedIndex = runDenseModelNNAPI(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "DenseNNAPI..."
                        val accuracy = runDenseModelNNAPITestSet() * 100
                        binding.resultText.text = "DenseNNAPI: ${formater.format(accuracy)} %"
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "DenseNNAPI..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = runDenseModelNNAPI(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "DenseNNAPI: ${formater.format(correctPredictions / 100f)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "DenseNNAPI"))
            }
        }

        with (binding.classify4in16Btn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        var predictedIndex : Int = 0
                        for (i in 0 until _runs)
                        {
                            predictedIndex = run4in16model(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "4in16Sparse..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = run4in16model(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "4in16Sparse: ${formater.format(correctPredictions / 100f)} %"
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "4in16Sparse..."
                        val accuracy = run4in16modelTestSet() * 100
                        binding.resultText.text = "4in16Sparse: ${formater.format(accuracy)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "4in16Sparse"))
            }
        }

        with (binding.classify2in16Btn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        var predictedIndex : Int = 0
                        for (i in 0 until _runs)
                        {
                            predictedIndex = run2in16model(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "2in16Sparse..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = run2in16model(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "2in16Sparse: ${formater.format(correctPredictions / 100f)} %"
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "2in16Sparse..."
                        val accuracy = run2in16modelTestSet() * 100
                        binding.resultText.text = "2in16Sparse: ${formater.format(accuracy)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "2in16Sparse"))
            }
        }

        with (binding.classify4in16ThreadsBtn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        var predictedIndex : Int = 0
                        for (i in 0 until _runs)
                        {
                            predictedIndex = run4in16modelThreads(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "4in16SparseThreads..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = run4in16modelThreads(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "4in16SparseThreads: ${formater.format(correctPredictions / 100f)} %"
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "4in16SparseThreads..."
                        val accuracy = run4in16modelThreadsTestSet() * 100
                        binding.resultText.text = "4in16SparseThreads: ${formater.format(accuracy)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "4in16SparseThreads"))
            }
        }

        with (binding.classify2in16ThreadsBtn) {
            setOnClickListener {
                val pixelArray = getPixelArray(rescaleAndConvertToMonochrome(binding.digitDrawView.getBitmap(), 32, 32))
                val formater = DecimalFormat("#.###")

                val elapsed = measureNanoTime {
                    if (_mode == Modes.SAMPLES) {
                        var predictedIndex : Int = 0
                        for (i in 0 until _runs)
                        {
                            predictedIndex = run2in16modelThreads(pixelArray)
                        }
                        binding.resultText.text = predictedIndex.toString()
                    }
                    else if (_mode == Modes.SAMPLED_TEST_SET) {
                        binding.resultText.text = "2in16SparseThreads..."
                        var correctPredictions : Int = 0
                        for (i in X_test.indices)
                        {
                            val predictedIndex = run2in16modelThreads(X_test[i])
                            if (predictedIndex == y_test[i][0].toInt()) {
                                correctPredictions++
                            }
                        }
                        binding.resultText.text = "2in16SparseThreads: ${formater.format(correctPredictions / 100f)} %"
                    }
                    else if (_mode == Modes.TEST_SET) {
                        binding.resultText.text = "2in16SparseThreads..."
                        val accuracy = run2in16modelThreadsTestSet() * 100
                        binding.resultText.text = "2in16SparseThreads: ${formater.format(accuracy)} %"
                    }
                }

                binding.performanceTotalText.text = "total: ${formater.format(elapsed / 1000_000.0f)} ms"

                if (_mode == Modes.SAMPLES) {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / _runs)} ms"
                }
                else {
                    binding.performanceAverageText.text = "average: ${formater.format(elapsed / 1000_000.0f / 10_000)} ms"
                }

                _measurements.add(Measurement(elapsed / 1000_000.0f, _runs, _mode, "2in16SparseThreads"))
            }
        }

        with (binding.digitDrawView) {
            setStrokeWidth(70f)
            setColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
        }

        with (binding.runsInput) {
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence, start: Int, count: Int, after: Int) { }

                override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) { }

                override fun afterTextChanged(s: Editable)
                {
                    _runs = if (s.isNotEmpty()) s.toString().toInt() else 0
                }
            })
        }

        with (binding.modeBtn) {
            setOnClickListener {
                if (_mode == Modes.SAMPLES) {
                    _mode = Modes.TEST_SET
                    text = "Matrix test set"
                    setBackgroundColor(Color.parseColor("#00FF00"))
                    binding.classifyTFOptimizedBtn.isEnabled = false
                    binding.classifyTFDefaultBtn.isEnabled = false
                    binding.classifyTFQuantizedBtn.isEnabled = false
                }
                else if (_mode == Modes.TEST_SET) {
                    _mode = Modes.SAMPLED_TEST_SET
                    text = "Sampled test set"
                    setBackgroundColor(Color.parseColor("#0000FF"))
                    binding.classifyTFOptimizedBtn.isEnabled = true
                    binding.classifyTFDefaultBtn.isEnabled = true
                    binding.classifyTFQuantizedBtn.isEnabled = true
                }
                else if (_mode == Modes.SAMPLED_TEST_SET) {
                    _mode = Modes.SAMPLES
                    text = "Screen runs"
                    setBackgroundColor(Color.parseColor("#FF0000"))
                }
            }

            text = "Screen runs"
            setBackgroundColor(Color.parseColor("#FF0000"))
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        quantizedModel.close()
        defaultModel.close()
        optimizedModel.close()
    }

    external fun loadModels(): Unit
    external fun runDenseThreadsSample(sample: FloatArray): Int
    external fun runDenseThreadsTestSet() : Float
    external fun runDenseModel(sample: FloatArray): Int
    external fun run4in16model(sample: FloatArray): Int
    external fun run4in16modelTestSet(): Float
    external fun run2in16model(sample: FloatArray): Int
    external fun run2in16modelTestSet(): Float
    external fun run4in16modelThreads(sample: FloatArray): Int
    external fun run4in16modelThreadsTestSet(): Float
    external fun run2in16modelThreads(sample: FloatArray): Int
    external fun run2in16modelThreadsTestSet(): Float
    external fun runDenseModelNNAPI(sample: FloatArray): Int
    external fun runDenseModelTestSet(): Float
    external fun runDenseModelNNAPITestSet(): Float

    companion object {
        // Used to load the 'sparseandroidml' library on application startup.
        init {
            System.loadLibrary("sparseandroidml")
        }
    }
}