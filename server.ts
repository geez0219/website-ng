/***************************************************************************************************
 * Load `$localize` onto the global scope - used if i18n tags appear in Angular templates.
 */
import { APP_BASE_HREF } from "@angular/common";
import "@angular/localize/init";
import { ngExpressEngine } from "@nguniversal/express-engine";
import * as express from "express";
import * as fs from "fs";
import { existsSync } from "fs";
import * as nodemailer from "nodemailer";
import { join } from "path";
import { tsParticles } from "tsparticles";
import "zone.js/dist/zone-node";
import { AppServerModule } from "./src/main.server";

const domino = require("domino");
const template = fs
  .readFileSync(join("dist", "fe-website", "browser", "index.html"))
  .toString();
const win = domino.createWindow(template);

global["window"] = win;
global["document"] = win.document;
global["tsParticles"] = tsParticles;

// The Express app is exported so that it can be used by serverless Functions.
export function app() {
  const server = express();
  const distFolder = join(process.cwd(), "dist/fe-website/browser");
  const indexHtml = existsSync(join(distFolder, "index.original.html"))
    ? "index.original.html"
    : "index";

  server.use(function (req, res, next) {
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    next();
  });

  server.use(express.urlencoded()); // add to handle slack form

  // Our Universal express-engine (found @ https://github.com/angular/universal/tree/master/modules/express-engine)
  server.engine(
    "html",
    ngExpressEngine({
      bootstrap: AppServerModule,
    })
  );

  server.set("view engine", "html");
  server.set("views", distFolder);

  // Serve static files from /browser
  server.get(
    "*.*",
    express.static(distFolder, {
      etag: false,
    })
  );

  // All regular routes use the Universal engine
  server.get("*", (req, res) => {
    res.render(indexHtml, {
      req,
      providers: [{ provide: APP_BASE_HREF, useValue: req.baseUrl }],
    });
  });

  //add to handle slack form
  server.post("/submit-form", (req, res) => {
    let content = req.body;
    sendMail(content, (info) => {
      // parse the 250 code from info.response
      res.redirect(
        "/community?slackEmailResponse=" + info.response.split(" ")[0]
      );
    });
  });

  return server;
}

function run() {
  const port = process.env.PORT || 4000;

  // Start up the Node server
  const server = app();
  server.listen(port, () => {
    console.log(`Node Express server listening on http://localhost:${port}`);
  });
}

// Webpack will replace 'require' with '__webpack_require__'
// '__non_webpack_require__' is a proxy to Node 'require'
// The below code is to ensure that the server is run only when not requiring the bundle.
declare const __non_webpack_require__: NodeRequire;
const mainModule = __non_webpack_require__.main;
const moduleFilename = (mainModule && mainModule.filename) || "";
if (moduleFilename === __filename || moduleFilename.includes("iisnode")) {
  run();
}

async function sendMail(content, callback) {
  // create reusable transporter object using the default SMTP transport
  let transporter = nodemailer.createTransport({
    host: "email-smtp.us-west-2.amazonaws.com",
    port: 587,
    secure: false, // true for 465, false for other ports
    auth: {
      user: "AKIAUPT4I4DGZLRPS55L",
      pass: "BJbE8q27Z6WmeskYId5aF+nvwB5W9HCzI+p/WuO08Xpt",
    },
  });

  let mailOptions = {
    from: '"slack channel join request" <fastestimator.dev@gmail.com>', // sender address
    to: "fastestimator.dev@gmail.com", // list of receivers
    subject: "slack channel join request from " + content.email, // Subject line
    html: `<p> email: ${content.email} </p>
          <ul>
            <li> What is your primary purpose to join? <br>
                ${content.purpose}, ${content.otherPurpose}
            </li>

            <li> Have you contributed to any open source ML based framework? <br>
                ${content.exp}, ${content.otherExp}
          </ul>
      `,
  };

  // send mail with defined transport object
  let info = await transporter.sendMail(mailOptions);
  callback(info);
}

export * from "./src/main.server";
