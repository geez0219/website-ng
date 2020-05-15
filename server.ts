/**
 * *** NOTE ON IMPORTING FROM ANGULAR AND NGUNIVERSAL IN THIS FILE ***
 *
 * If your application uses third-party dependencies, you'll need to
 * either use Webpack or the Angular CLI's `bundleDependencies` feature
 * in order to adequately package them for use on the server without a
 * node_modules directory.
 *
 * However, due to the nature of the CLI's `bundleDependencies`, importing
 * Angular in this file will create a different instance of Angular than
 * the version in the compiled application code. This leads to unavoidable
 * conflicts. Therefore, please do not explicitly import from @angular or
 * @nguniversal in this file. You can export any needed resources
 * from your application's main.server.ts file, as seen below with the
 * import for `ngExpressEngine`.
 */

import 'zone.js/dist/zone-node';
import * as cors from 'cors';
import * as express from 'express';
import {join} from 'path';
import * as nodemailer from 'nodemailer'
// import {tsParticles} from 'tsparticles'

// fix window is not defined error
const domino = require('domino');
const fs = require('fs');
const template = fs.readFileSync(join('dist', 'browser', 'index.html')).toString();
const win = domino.createWindow(template);
global['window'] = win;
global['document'] = win.document;
global['tsParticles'] = require("tsparticles");


// Express server
const app = express();
const PORT = process.env.PORT || 4000;
const DIST_FOLDER = join(process.cwd(), 'dist/browser');

// nodemailer: to send email 

// * NOTE :: leave this as require() since this file is built Dynamically from webpack
const {AppServerModuleNgFactory, LAZY_MODULE_MAP, ngExpressEngine, provideModuleMap} = require('./dist/server/main');

app.use(cors());
app.use(express.urlencoded()) // add to handle slack form 

// Our Universal express-engine (found @ https://github.com/angular/universal/tree/master/modules/express-engine)
app.engine('html', ngExpressEngine({
  bootstrap: AppServerModuleNgFactory,
  providers: [
    provideModuleMap(LAZY_MODULE_MAP)
  ]
}));

app.set('view engine', 'html');
app.set('views', DIST_FOLDER);

//allow OPTIONS on just one resource
app.options('*.*', cors());

// add to handle slack form
app.post('/submit-form', (req, res) => {
  //...
  // console.log(req.body.username);
  // console.log(req.body.username2);
  // console.log("hello");
  // res.end()

  console.log("request came");
  let content = req.body;
  // res.end();

  sendMail(content, info => {
    // parse the 250 code from info.response
    res.redirect('/community?slackEmailResponse='+info.response.split(' ')[0]);
  });
})

// Example Express Rest API endpoints
// app.get('/api/**', (req, res) => { });
// Serve static files from /browser
app.get('*.*', cors(), express.static(DIST_FOLDER, {
  maxAge: '1y'
}));

// All regular routes use the Universal engine
app.get('*', (req, res) => {
  res.render('index', { req });
});

// Start up the Node server
app.listen(PORT, () => {
  console.log(`Node Express server listening on http://localhost:${PORT}`);
});


async function sendMail(content, callback) {
  // create reusable transporter object using the default SMTP transport
  let transporter = nodemailer.createTransport({
    host: "email-smtp.us-west-2.amazonaws.com",
    port: 587,
    secure: false, // true for 465, false for other ports
    auth: {
      user: "AKIAUPT4I4DGZLRPS55L",
      pass: "BJbE8q27Z6WmeskYId5aF+nvwB5W9HCzI+p/WuO08Xpt"
    }
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
      `
  };

  // send mail with defined transport object
  let info = await transporter.sendMail(mailOptions);
  callback(info);
}