package com.example.faceverify

import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.core.Camera
import androidx.core.content.ContextCompat
import java.io.File

@Composable
fun CameraScreen(
    onCaptured: (File) -> Unit,
    onError: (String) -> Unit,
    onCancel: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var capture by remember {
        mutableStateOf<ImageCapture?>(null)
    }

    var camera by remember {
        mutableStateOf<Camera?>(null)
    }

    var provider by remember {
        mutableStateOf<ProcessCameraProvider?>(null)
    }

    var ready by remember { mutableStateOf(false) }
    var takingPhoto by remember { mutableStateOf(false) }

    DisposableEffect(provider) {
        onDispose {
            provider?.unbindAll()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            "Capture live face",
            style = MaterialTheme.typography.headlineSmall
        )

        Text("Look at the camera in a well-lit area.")

        AndroidView(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            factory = { viewContext ->
                PreviewView(viewContext).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )

                    scaleType = PreviewView.ScaleType.FILL_CENTER

                    val future =
                        ProcessCameraProvider.getInstance(viewContext)

                    future.addListener({
                        try {
                            val cameraProvider = future.get()
                            provider = cameraProvider

                            val preview = Preview.Builder().build()
                            preview.surfaceProvider = surfaceProvider

                            val imageCapture = ImageCapture.Builder()
                                .setCaptureMode(
                                    ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
                                )
                                .build()

                            cameraProvider.unbindAll()

                            camera = cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_FRONT_CAMERA,
                                preview,
                                imageCapture
                            )

                            capture = imageCapture
                            ready = true
                        } catch (e: Exception) {
                            onError(
                                e.message ?: "Unable to start front camera."
                            )
                        }
                    }, ContextCompat.getMainExecutor(viewContext))
                }
            }
        )

        Button(
            modifier = Modifier.fillMaxWidth(),
            enabled = ready && !takingPhoto,
            onClick = {
                val imageCapture = capture ?: return@Button

                takingPhoto = true

                val photoFile = File(
                    context.cacheDir,
                    "live_${System.currentTimeMillis()}.jpg"
                )

                val options = ImageCapture.OutputFileOptions.Builder(
                    photoFile
                ).build()

                imageCapture.takePicture(
                    options,
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageSavedCallback {

                        override fun onImageSaved(
                            output: ImageCapture.OutputFileResults
                        ) {
                            takingPhoto = false
                            onCaptured(photoFile)
                        }

                        override fun onError(
                            exception: ImageCaptureException
                        ) {
                            takingPhoto = false
                            photoFile.delete()

                            onError(
                                exception.message ?: "Photo capture failed."
                            )
                        }
                    }
                )
            }
        ) {
            Text(
                if (takingPhoto) "Capturing..." else "Capture face"
            )
        }

        OutlinedButton(
            modifier = Modifier.fillMaxWidth(),
            onClick = onCancel
        ) {
            Text("Cancel")
        }
    }
}