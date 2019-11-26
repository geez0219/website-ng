import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { MarkdownModule } from 'ngx-markdown';
import { ClipboardModule } from 'ngx-clipboard';
import { MatNativeDateModule, MatCardModule, MatDialogModule, MatSnackBarModule, MatTreeModule, MatToolbarModule, MatIconModule, MatSidenavModule, MatListModule, MatButtonModule } from  '@angular/material';

import { WINDOW_PROVIDERS } from './window-provider/window-provider.component';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { DialogComponent } from './dialog/dialog.component';
import { NavbarComponent } from './navbar/navbar.component';
import { GettingStartedComponent } from './getting-started/getting-started.component';
import { TutorialComponent } from './tutorial/tutorial.component';
import { ApiComponent } from './api/api.component';
import { TocComponent } from './toc/toc.component';
import { InstallComponent } from './install/install.component';
import { CommunityComponent } from './community/community.component';
import { PageNotFoundComponent } from './page-not-found/page-not-found.component';
import { SnackbarComponent } from './snackbar/snackbar.component';
import { ExampleComponent } from './example/example.component';
import { SearchComponent } from './search/search.component';
import { FooterComponent } from './footer/footer.component';
 
@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    GettingStartedComponent,
    TutorialComponent,
    ApiComponent,
    TocComponent,
    InstallComponent,
    CommunityComponent,
    SnackbarComponent,
    PageNotFoundComponent,
    ExampleComponent,
    DialogComponent,
    SearchComponent,
    FooterComponent,
  ],
  imports: [
    BrowserModule.withServerTransition({ appId: 'serverApp' }),
    AppRoutingModule,
    HttpClientModule,
    FormsModule,

    NgbModule,
    MarkdownModule.forRoot(),
    BrowserAnimationsModule,

    MatToolbarModule,
    MatSidenavModule,
    MatListModule,
    MatButtonModule,
    MatIconModule,
    MatNativeDateModule,
    MatTreeModule,
    ClipboardModule,
    MatSnackBarModule,
    MatDialogModule,
    MatCardModule
  ],
  providers: [WINDOW_PROVIDERS],
  bootstrap: [AppComponent],
  entryComponents: [SnackbarComponent, DialogComponent]
})
export class AppModule { }
