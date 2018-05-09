package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

android_binary(
    name = "tfmobile",
    srcs = glob(["src/**/*.java"]),
    assets = [
        "//assets:graph.pb",
        "//assets:labels.txt",
    ],
    assets_dir = "",
    custom_package = "org.tensorflow.demo",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = [
        ".pb",
    ],
    resource_files = glob(["res/**"]),
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        "@androidsdk//com.android.support:support-v13-25.2.0",
        "@androidsdk//com.android.support:support-v4-25.2.0",
        "@tensorflow_android//aar",
    ],
)

exports_files(["jcenter.xml"])
