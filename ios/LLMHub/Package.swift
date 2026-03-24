// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "LLMHub",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "LLMHub",
            targets: ["LLMHub"]
        ),
    ],
    dependencies: [
        .package(path: "../runanywhere-sdks-latest")
    ],
    targets: [
        .target(
            name: "LLMHub",
            dependencies: [
                .product(name: "RunAnywhere", package: "runanywhere-sdks"),
                .product(name: "RunAnywhereLlamaCPP", package: "runanywhere-sdks")
            ],
            exclude: [
                "check_strings.py"
            ],
            resources: [
                .process("Icon.png"),
                .process("en.lproj"),
                .process("ar.lproj"),
                .process("de.lproj"),
                .process("es.lproj"),
                .process("fa.lproj"),
                .process("fr.lproj"),
                .process("he.lproj"),
                .process("id.lproj"),
                .process("it.lproj"),
                .process("ja.lproj"),
                .process("ko.lproj"),
                .process("pl.lproj"),
                .process("pt.lproj"),
                .process("ru.lproj"),
                .process("tr.lproj"),
                .process("uk.lproj")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
    ]
)
