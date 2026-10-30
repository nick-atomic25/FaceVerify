package com.example.faceverify

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

data class VerificationResult(
    val similarity: Float,
    val threshold: Float,
    val matched: Boolean
)

class FaceEngine(context: Context) {

    companion object {
        private const val INPUT_SIZE = 112
        const val DEFAULT_THRESHOLD = 0.80f
    }

    private val appContext = context.applicationContext

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(
                FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE
            )
            .build()
    )

    private val modelDelegate = lazy {
        val modelBuffer = appContext.assets
            .open("mobile_face_net.tflite")
            .use { input ->
                val bytes = input.readBytes()
                ByteBuffer.allocateDirect(bytes.size)
                    .order(ByteOrder.nativeOrder())
                    .apply {
                        put(bytes)
                        rewind()
                    }
            }

        Interpreter(
            modelBuffer,
            Interpreter.Options().apply {
                setNumThreads(4)
            }
        ).also { interpreter ->
            val inputShape =
                interpreter.getInputTensor(0).shape()

            require(
                inputShape.contentEquals(
                    intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3)
                )
            ) {
                "Model input must be [1,112,112,3]."
            }

            require(
                interpreter.getInputTensor(0).dataType() ==
                    org.tensorflow.lite.DataType.FLOAT32
            ) {
                "Model input must be FLOAT32."
            }

            require(
                interpreter.getOutputTensor(0).dataType() ==
                    org.tensorflow.lite.DataType.FLOAT32
            ) {
                "Model output must be FLOAT32."
            }

            require(
                interpreter.getOutputTensor(0).numElements() > 0
            ) {
                "Model has an invalid output."
            }
        }
    }

    private val interpreter: Interpreter
        get() = modelDelegate.value

    fun loadPhoto(uri: Uri): Bitmap {
        val original = appContext.contentResolver
            .openInputStream(uri)
            .use { stream ->
                requireNotNull(stream) {
                    "Cannot open selected image."
                }

                BitmapFactory.decodeStream(stream)
            } ?: error("Unsupported image.")

        val orientation = appContext.contentResolver
            .openInputStream(uri)
            .use { stream ->
                if (stream == null) {
                    ExifInterface.ORIENTATION_NORMAL
                } else {
                    ExifInterface(stream).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                }
            }

        val matrix = Matrix()

        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 ->
                matrix.postRotate(90f)

            ExifInterface.ORIENTATION_ROTATE_180 ->
                matrix.postRotate(180f)

            ExifInterface.ORIENTATION_ROTATE_270 ->
                matrix.postRotate(270f)

            ExifInterface.ORIENTATION_FLIP_HORIZONTAL ->
                matrix.postScale(-1f, 1f)

            ExifInterface.ORIENTATION_FLIP_VERTICAL ->
                matrix.postScale(1f, -1f)

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postScale(-1f, 1f)
                matrix.postRotate(90f)
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postScale(-1f, 1f)
                matrix.postRotate(270f)
            }
        }

        if (
            orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED
        ) {
            return original
        }

        return Bitmap.createBitmap(
            original,
            0,
            0,
            original.width,
            original.height,
            matrix,
            true
        ).also { corrected ->
            if (corrected !== original) {
                original.recycle()
            }
        }
    }

    // Call from a background coroutine.
    fun embedding(bitmap: Bitmap): FloatArray {
        val image = InputImage.fromBitmap(bitmap, 0)

        val faces = Tasks.await(detector.process(image))

        require(faces.size == 1) {
            when (faces.size) {
                0 -> "No face detected. Try better lighting."
                else -> "Multiple faces detected. Use a photo with one face."
            }
        }

        val bounds = faces.single().boundingBox

        require(
            bounds.width() >= 80 &&
            bounds.height() >= 80
        ) {
            "Face is too small. Choose a closer photo."
        }

        val marginX = (bounds.width() * 0.15f).toInt()
        val marginY = (bounds.height() * 0.15f).toInt()

        val left = (bounds.left - marginX).coerceAtLeast(0)
        val top = (bounds.top - marginY).coerceAtLeast(0)
        val right = (bounds.right + marginX)
            .coerceAtMost(bitmap.width)
        val bottom = (bounds.bottom + marginY)
            .coerceAtMost(bitmap.height)

        require(right > left && bottom > top) {
            "Invalid face crop."
        }

        val crop = Bitmap.createBitmap(
            bitmap,
            left,
            top,
            right - left,
            bottom - top
        )

        val resized = Bitmap.createScaledBitmap(
            crop,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )

        if (resized !== crop) crop.recycle()

        try {
            val input = ByteBuffer.allocateDirect(
                INPUT_SIZE * INPUT_SIZE * 3 * 4
            ).order(ByteOrder.nativeOrder())

            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val pixel = resized.getPixel(x, y)

                    input.putFloat(
                        (((pixel shr 16) and 0xFF) / 127.5f) - 1f
                    )
                    input.putFloat(
                        (((pixel shr 8) and 0xFF) / 127.5f) - 1f
                    )
                    input.putFloat(
                        ((pixel and 0xFF) / 127.5f) - 1f
                    )
                }
            }

            input.rewind()

            val outputSize =
                interpreter.getOutputTensor(0).numElements()

            val output = Array(1) {
                FloatArray(outputSize)
            }

            interpreter.run(input, output)

            val vector = output[0]

            val norm = sqrt(
                vector.sumOf {
                    it.toDouble() * it
                }
            ).toFloat()

            require(norm.isFinite() && norm > 1e-8f) {
                "Invalid face embedding."
            }

            return FloatArray(vector.size) {
                vector[it] / norm
            }
        } finally {
            resized.recycle()
        }
    }

    fun compare(
        reference: FloatArray,
        live: FloatArray,
        threshold: Float
    ): VerificationResult {
        require(reference.isNotEmpty()) {
            "Reference embedding is empty."
        }

        require(reference.size == live.size) {
            "Embedding sizes differ. Check your recognition model."
        }

        require(threshold in -1f..1f) {
            "Threshold must be between -1 and 1."
        }

        val similarity = reference.indices.sumOf {
            reference[it].toDouble() * live[it]
        }.toFloat()

        return VerificationResult(
            similarity = similarity,
            threshold = threshold,
            matched = similarity >= threshold
        )
    }

    fun close() {
        detector.close()

        if (modelDelegate.isInitialized()) {
            interpreter.close()
        }
    }
}