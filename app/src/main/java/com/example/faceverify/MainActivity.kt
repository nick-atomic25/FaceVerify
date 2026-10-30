package com.example.faceverify

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                FaceVerifyApp()
            }
        }
    }
}

@Composable
private fun FaceVerifyApp() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val engine = remember { FaceEngine(context) }

    var referenceEmbedding by remember {
        mutableStateOf<FloatArray?>(null)
    }

    var referenceBitmap by remember {
        mutableStateOf<Bitmap?>(null)
    }

    var threshold by remember {
        mutableFloatStateOf(FaceEngine.DEFAULT_THRESHOLD)
    }

    var result by remember {
        mutableStateOf<VerificationResult?>(null)
    }

    var busy by remember { mutableStateOf(false) }
    var cameraOpen by remember { mutableStateOf(false) }

    var cameraAllowed by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    var message by remember { mutableStateOf<String?>(null) }

    DisposableEffect(engine) {
        onDispose {
            engine.close()
            referenceBitmap?.recycle()
        }
    }

    val photoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->

        if (uri != null) {
            scope.launch {
                busy = true
                result = null
                message = null

                try {
                    val bitmap = withContext(Dispatchers.IO) {
                        engine.loadPhoto(uri)
                    }

                    val embedding = try {
                        withContext(Dispatchers.Default) {
                            engine.embedding(bitmap)
                        }
                    } catch (e: Exception) {
                        bitmap.recycle()
                        throw e
                    }

                    referenceBitmap?.recycle()
                    referenceBitmap = bitmap
                    referenceEmbedding = embedding
                    message = "Reference face is ready."
                } catch (e: Exception) {
                    referenceEmbedding = null
                    message = e.message
                        ?: "Unable to process reference photo."
                } finally {
                    busy = false
                }
            }
        }
    }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        cameraAllowed = granted

        if (granted) {
            cameraOpen = true
        } else {
            message = "Camera permission is required."
        }
    }

    if (cameraOpen) {
        CameraScreen(
            onCaptured = { photoFile ->
                cameraOpen = false

                scope.launch {
                    busy = true
                    result = null
                    message = "Analyzing live photo..."

                    try {
                        val bitmap = withContext(Dispatchers.IO) {
                            engine.loadPhoto(Uri.fromFile(photoFile))
                        }

                        try {
                            val liveEmbedding = withContext(
                                Dispatchers.Default
                            ) {
                                engine.embedding(bitmap)
                            }

                            val reference = referenceEmbedding
                                ?: error("Select a reference photo first.")

                            result = engine.compare(
                                reference = reference,
                                live = liveEmbedding,
                                threshold = threshold
                            )

                            message = null
                        } finally {
                            bitmap.recycle()
                        }
                    } catch (e: Exception) {
                        message = e.message
                            ?: "Face verification failed."
                    } finally {
                        photoFile.delete()
                        busy = false
                    }
                }
            },
            onError = { error ->
                cameraOpen = false
                message = error
            },
            onCancel = {
                cameraOpen = false
            }
        )
    } else {
        Scaffold { padding ->

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .verticalScroll(rememberScrollState())
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {

                Text(
                    "Face Verify",
                    style = MaterialTheme.typography.headlineLarge
                )

                Text(
                    "Compare a stored face with a live camera capture. " +
                        "Photos are processed locally."
                )

                Button(
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !busy,
                    onClick = {
                        photoPicker.launch("image/*")
                    }
                ) {
                    Text("1. Select reference photo")
                }

                referenceBitmap?.let { bitmap ->
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Reference photo",
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 240.dp)
                    )
                }

                Text(
                    if (referenceEmbedding == null) {
                        "No reference photo selected"
                    } else {
                        "Reference face ready"
                    }
                )

                HorizontalDivider()

                Text(
                    "Similarity threshold: %.2f".format(threshold)
                )

                Slider(
                    value = threshold,
                    onValueChange = { threshold = it },
                    valueRange = 0.50f..0.95f,
                    steps = 44,
                    enabled = !busy
                )

                Text(
                    "Higher thresholds require greater similarity. " +
                        "The correct threshold depends on the model."
                )

                Button(
                    modifier = Modifier.fillMaxWidth(),
                    enabled = referenceEmbedding != null && !busy,
                    onClick = {
                        if (cameraAllowed) {
                            cameraOpen = true
                        } else {
                            cameraPermission.launch(
                                Manifest.permission.CAMERA
                            )
                        }
                    }
                ) {
                    Text("2. Capture live face")
                }

                if (busy) {
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                message?.let {
                    Text(
                        text = it,
                        color = MaterialTheme.colorScheme.secondary
                    )
                }

                result?.let { verification ->

                    HorizontalDivider()

                    Text(
                        text = if (verification.matched) {
                            "MATCH"
                        } else {
                            "NO MATCH"
                        },
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = if (verification.matched) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.error
                        }
                    )

                    Text(
                        "Cosine similarity: %.4f".format(
                            verification.similarity
                        )
                    )

                    Text(
                        "Threshold: %.2f".format(
                            verification.threshold
                        )
                    )
                }
            }
        }
    }
}