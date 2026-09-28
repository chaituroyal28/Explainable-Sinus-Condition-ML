/* =========================================================
   SinusPredict AI
   Global JavaScript
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* -----------------------------------------------------
       Theme
       ----------------------------------------------------- */

    const themeToggle = document.getElementById("themeToggle");

    const savedTheme = localStorage.getItem("sinuspredict-theme");

    if (savedTheme) {
        document.documentElement.setAttribute(
            "data-theme",
            savedTheme
        );
    }


    function updateThemeIcon() {

        if (!themeToggle) {
            return;
        }

        const currentTheme =
            document.documentElement.getAttribute("data-theme");

        themeToggle.textContent =
            currentTheme === "dark" ? "☀" : "◐";
    }


    updateThemeIcon();


    if (themeToggle) {

        themeToggle.addEventListener("click", () => {

            const currentTheme =
                document.documentElement.getAttribute("data-theme");

            const newTheme =
                currentTheme === "dark"
                    ? "light"
                    : "dark";

            document.documentElement.setAttribute(
                "data-theme",
                newTheme
            );

            localStorage.setItem(
                "sinuspredict-theme",
                newTheme
            );

            updateThemeIcon();

        });

    }


    /* -----------------------------------------------------
       Page entrance animation
       ----------------------------------------------------- */

    document.body.classList.add("page-loaded");


    /* -----------------------------------------------------
       Button interaction
       ----------------------------------------------------- */

    const buttons =
        document.querySelectorAll(".btn");

    buttons.forEach((button) => {

        button.addEventListener("click", () => {

            button.classList.add("clicked");

            setTimeout(() => {
                button.classList.remove("clicked");
            }, 250);

        });

    });

});

const downloadCsv =
    document.getElementById("downloadCsv");

if (downloadCsv) {

    downloadCsv.addEventListener(
        "click",
        () => {

            const rows = [];

            rows.push([
                "Field",
                "Value"
            ]);

            rows.push([
                "Prediction",
                result.prediction
            ]);

            rows.push([
                "Model Probability",
                result.probability + "%"
            ]);

            Object.entries(result.inputs)
                .forEach(([key, value]) => {

                    rows.push([
                        key,
                        value
                    ]);

                });


            const csv = rows
                .map(row =>
                    row
                        .map(value =>
                            `"${String(value)
                                .replace(/"/g, '""')}"`
                        )
                        .join(",")
                )
                .join("\n");


            const blob =
                new Blob(
                    [csv],
                    { type: "text/csv" }
                );


            const url =
                URL.createObjectURL(blob);


            const link =
                document.createElement("a");

            link.href = url;

            link.download =
                "sinuspredict-result.csv";

            link.click();

            URL.revokeObjectURL(url);

        }
    );

}